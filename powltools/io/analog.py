"""Representation of and interaction with analog data in pOwl files

Analog data in pOwl files is typically distributed across trials and saved in
snippets corresponding to each trial in a path like this example:

    /trials/0/traces/5/data_array

Where the individual parts have the following meanings:

-   "trials" is the standard name for the HDF5 group containing all trials
-   "0" is the trial_index,
-   "traces" is the signal_key, indicating what data is contained
-   "5" is the channel_number, corresponding to TDT's ADC channels
-   "data_array" is pOwl's name for HDF5 datasets with analog data

Next to "trials" in the top-level of a pOwl file, there may be another group
called "pauses" which is similarly structured. Numeric keys of each pause
indicate _before_ which trial the pause occured. Note that the entire pauses
group can be missing, and even when it exists, each pause may or may not be
present.

"""

from __future__ import annotations
from functools import cached_property, partial
from typing import Any, Optional, Callable
import numpy as np
import numpy.typing as npt
import h5py
import scipy

from powltools.filters.helpers import chain_filters
from .file import POwlFile, PauseNotFound
from .parameters import save_params

SignalType = npt.NDArray[np.float64]
FilterType = Callable[[SignalType], SignalType]
StartStopDict = dict[int, tuple[int, int]]
SnippetDict = dict[int, SignalType]


class AnalogSignal:
    """Representation of continuous analog signal"""

    def __init__(
        self,
        powlfile: POwlFile,
        channel_number: int,
        signal_key: str = "traces",
    ) -> None:
        self.powlfile = powlfile
        self.channel_number = channel_number
        self.signal_key = signal_key
        self._trials_start_stop: StartStopDict = {}
        self._pauses_start_stop: StartStopDict = {}
        self._loaded = False

    @cached_property
    def group_key(self) -> str:
        return f"{self.signal_key}/{self.channel_number}"

    @cached_property
    def data_key(self) -> str:
        return f"{self.group_key}/data_array"

    @property
    def trials_start_stop(self) -> StartStopDict:
        if not self._loaded:
            raise AnalogSignalNotRead()
        return self._trials_start_stop

    @property
    def pauses_start_stop(self) -> StartStopDict:
        if not self._loaded:
            raise AnalogSignalNotRead()
        return self._pauses_start_stop

    @property
    def continuous_signal(self) -> SignalType:
        if not self._loaded:
            raise AnalogSignalNotRead()
        return self._continuous_signal

    def _get_snippet(self, trial_or_pause: h5py.Group) -> SignalType:
        return np.array(trial_or_pause[self.data_key])

    def load(self):
        """Loads the continuous, concatenated version of an analog signal
        across trials and pauses if there are any."""
        snippets: list[SignalType] = []
        current_index: int = 0
        self._trials_start_stop.clear()
        self._pauses_start_stop.clear()

        with self.powlfile:
            for trial_index, trial_group in self.powlfile.trials():
                try:
                    pause_group = self.powlfile.get_pause(trial_index)
                except PauseNotFound:
                    pass
                else:
                    pause_snippet = self._get_snippet(pause_group)
                    snippets.append(pause_snippet)
                    start, stop = current_index, current_index + pause_snippet.size
                    self._pauses_start_stop[trial_index] = (start, stop)
                    current_index = stop
                trial_snippet = self._get_snippet(trial_group)
                snippets.append(trial_snippet)
                start, stop = current_index, current_index + trial_snippet.size
                self._trials_start_stop[trial_index] = (start, stop)
                current_index = stop
            self._continuous_signal = np.hstack(snippets)
            self._loaded = True

    def _put_snippet(
        self,
        trial_or_pause: h5py.Group,
        start_stop: tuple[int, int],
    ) -> h5py.Group:
        data_group = trial_or_pause.create_group(self.group_key)
        data_group.create_dataset(
            name="data_array",
            data=self.continuous_signal[slice(*start_stop)],
        )
        return data_group

    def save(self, parameters: Optional[dict[str, Any]] = None):
        with self.powlfile:
            for trial_index, trial_group in self.powlfile.trials():
                data_group = self._put_snippet(
                    trial_group, self.trials_start_stop[trial_index]
                )
                if parameters:
                    save_params(parameters, data_group)
            for trial_index in sorted(self.pauses_start_stop.keys(), key=int):
                try:
                    pause_group = self.powlfile.get_pause(trial_index)
                except PauseNotFound:
                    continue
                pause_data_group = self._put_snippet(
                    pause_group, self.pauses_start_stop[trial_index]
                )
                if parameters:
                    save_params(parameters, pause_data_group)

    def copy_filtered(
        self,
        filter_func: FilterType,
        signal_key: str,
        astype: np.dtype | None = None,
    ) -> AnalogSignal:
        if not self._loaded:
            raise AnalogSignalNotRead(
                "Can not use copy_filtered() before signal is loaded."
            )
        copy_instance = AnalogSignal(
            self.powlfile,
            channel_number=self.channel_number,
            signal_key=signal_key,
        )
        copy_instance._pauses_start_stop.update(self.pauses_start_stop)
        copy_instance._trials_start_stop.update(self.trials_start_stop)
        copy_instance._continuous_signal = filter_func(self.continuous_signal)
        if astype is not None:
            copy_instance._continuous_signal = copy_instance._continuous_signal.astype(
                astype
            )
        copy_instance._loaded = True
        return copy_instance


class AnalogSnippets:
    """Representation of non-continuous analog signal"""

    def __init__(
        self,
        powlfile: POwlFile,
        channel_number: int,
        signal_key: str = "traces",
        pauses: bool = False,
    ) -> None:
        self.powlfile = powlfile
        self.channel_number = channel_number
        self.signal_key = signal_key
        self._trial_snippets: SnippetDict = {}
        self._pause_snippets: SnippetDict = {}
        self._pauses = pauses
        self._loaded = False

    @cached_property
    def group_key(self) -> str:
        return f"{self.signal_key}/{self.channel_number}"

    @cached_property
    def data_key(self) -> str:
        return f"{self.group_key}/data_array"

    @property
    def trial_snippets(self) -> SnippetDict:
        if not self._loaded:
            raise AnalogSignalNotRead()
        return self._trial_snippets

    @property
    def pause_snippets(self) -> SnippetDict:
        if not self._pauses:
            raise ValueError("AnalogSnippets without pauses")
        if not self._loaded:
            raise AnalogSignalNotRead()
        return self._pause_snippets

    def _get_snippet(self, trial_or_pause: h5py.Group) -> SignalType:
        return np.array(trial_or_pause[self.data_key])

    def load(self):
        self._trial_snippets.clear()
        self._pause_snippets.clear()
        with self.powlfile:
            for trial_index, trial_group in self.powlfile.trials():
                if self._pauses:
                    try:
                        pause_group = self.powlfile.get_pause(trial_index)
                    except PauseNotFound:
                        pass
                    else:
                        pause_snippet = self._get_snippet(pause_group)
                        self._pause_snippets[trial_index] = pause_snippet
                trial_snippet = self._get_snippet(trial_group)
                self._trial_snippets[trial_index] = trial_snippet
        self._loaded = True

    def _put_snippet(
        self,
        trial_or_pause: h5py.Group,
        snippet: SignalType,
    ) -> h5py.Group:
        data_group = trial_or_pause.create_group(self.group_key)
        data_group.create_dataset(
            name="data_array",
            data=snippet,
        )
        return data_group

    def save(self, parameters: Optional[dict[str, Any]] = None):
        with self.powlfile:
            for trial_index, trial_group in self.powlfile.trials():
                data_group = self._put_snippet(
                    trial_group, self.trial_snippets[trial_index]
                )
                if parameters:
                    save_params(parameters, data_group)
            if self._pauses:
                for trial_index in sorted(self.pause_snippets.keys()):
                    try:
                        pause_group = self.powlfile.get_pause(trial_index)
                    except PauseNotFound:
                        continue
                    pause_data_group = self._put_snippet(
                        pause_group, self.pause_snippets[trial_index]
                    )
                    if parameters:
                        save_params(parameters, pause_data_group)

    @classmethod
    def resampled_from_analog_signal(
        cls,
        analog_signal: AnalogSignal,
        signal_key: str,
        up: int = 1,
        down: int = 1,
        pauses: bool = False,
        astype: np.dtype | None = None,
    ) -> AnalogSnippets:
        new = cls(
            analog_signal.powlfile,
            channel_number=analog_signal.channel_number,
            signal_key=signal_key,
            pauses=pauses,
        )
        resample = partial(scipy.signal.resample_poly, up=up, down=down, padtype="line")
        if astype is not None:
            resample = chain_filters(resample, lambda signal: signal.astype(astype))
        with new.powlfile:
            for trial_index, start_stop in analog_signal.trials_start_stop.items():
                old_snippet = analog_signal.continuous_signal[slice(*start_stop)]
                new._trial_snippets[trial_index] = resample(old_snippet)
            if new._pauses:
                for trial_index, start_stop in analog_signal.pauses_start_stop.items():
                    old_snippet = analog_signal.continuous_signal[slice(*start_stop)]
                    new._pause_snippets[trial_index] = resample(old_snippet)
        new._loaded = True
        return new


class AnalogSignalNotRead(Exception):
    def __init__(
        self,
        message="Continuous data has not be read. Use .load() method first.",
    ) -> None:
        super().__init__(message)
