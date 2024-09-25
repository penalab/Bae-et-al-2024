from __future__ import annotations as _annotations
from functools import cached_property
import os
import re

import numpy as np
import numpy.typing as npt
from typing import Any, Callable, Iterable, Unpack, cast

from ..io.file import POwlFile
from ..io.parameters import get_params, GroupParams

TrialParams = dict[str, Any]
TrialParamFunc = Callable[[TrialParams], Any]
StimulusParams = dict[str, Any]
StimParamFunc = Callable[[StimulusParams], Any]
SpikeTimesType = npt.NDArray[np.float64]
SpikeTrainsType = dict[int, SpikeTimesType]
SpikeTimesFunc = Callable[..., Any]
LFPType = npt.NDArray[np.float64]
LFPSnippetsType = dict[int, LFPType]
LFPSnippetFunc = Callable[[LFPType], Any]


class StimulusNotFound(KeyError):
    def __init__(self, recording: Recording, stimulus_index: int) -> None:
        super().__init__()
        self.stimulus_index = stimulus_index
        self.recording = recording

    def __repr__(self) -> str:
        return f"StimulusNotFound(recording={self.recording!r}, stimulus_index={self.stimulus_index!r})"

    def __str__(self) -> str:
        return f"StimulusNotFound: Stimulus with index {self.stimulus_index} not found for {self.recording!r}."


def group_by_param(
    trial_values: npt.ArrayLike,
    trial_params: npt.ArrayLike,
) -> dict[float, np.ndarray]:
    """Group any values by common unique parameters

    For example, group response rates by stimulus level:

        # Input arrays:
        responses = rec.response_rates(channel_number=1, stimulus_index=0)
        levels = rec.aggregate_stim_params(stim_level, stimulus_index=0)
        # Grouping into a dict:
        responses_by_level = group_by_param(responses, levels)
        # For a rate level function with errorbars:
        plot_levels, resp_mean, resp_std = np.array(
            [
                [level, resp.mean(), resp.std()]
                for level, resp in responses_by_level.items()
            ]
        ).T
        plt.errorbar(plot_levels, resp_mean, yerr=resp_std)

    """
    trial_values = np.asarray(trial_values)
    trial_params = np.asarray(trial_params)
    uparameters = np.unique(trial_params, axis=0)
    if trial_params.ndim != 1:
        raise ValueError(
            "Use `group_by_multiparam()` for grouping by multiple parameters."
        )
    return {p: trial_values[trial_params == p] for p in uparameters}


def group_by_multiparam(
    trial_values: npt.ArrayLike,
    trial_params: npt.ArrayLike,
) -> dict[tuple[float, ...], np.ndarray]:
    """Group values by a set of parameters, such as the azi/ele position

    Similar to group_by_param, but the keys of the returned dict are tuples
    with the combination of parameters.

    Example of a color-coded spatial receptive field:

        # Input arrays:
        responses = rec.response_rates(channel_number=1, stimulus_index=0)
        positions = rec.aggregate_stim_params(stim_position, stimulus_index=0)
        # Grouping into a dict:
        responses_by_position = group_by_param(responses, positions)
        # For a spatial receptive field scatterplot:
        azimuths, elevations, resp_mean = np.array(
            [
                [azi, ele, resp.mean()]
                for (azi, ele), resp in responses_by_position.items()
            ]
        ).T
        plt.scatterplot(azimuths, elevations, c=resp_mean)

    """
    trial_values = np.asarray(trial_values)
    trial_params = np.asarray(trial_params)
    uparameters = np.unique(trial_params, axis=0)
    return {
        tuple(p): trial_values[(trial_params == p).all(axis=1)] for p in uparameters
    }


def stim_position(stim_params: StimulusParams) -> tuple[int, int]:
    return (stim_params["azi"], stim_params["ele"])


def stim_level(stim_params: StimulusParams) -> float:
    return stim_params["stim_func_kwargs"]["level"]


def stim_delay(stim_params: StimulusParams) -> float:
    return stim_params["delay_s"]


def stim_len(stim_params: StimulusParams) -> float:
    return stim_params["stim_func_factory_kwargs"]["stim_len_s"]


def stim_f_am(stim_params: StimulusParams) -> int:
    return stim_params["stim_func_factory_kwargs"]["modulation_frequency"]


class Recording:
    """Representation of one (Batch) Recording

    This class provides some low- to mid-level caching and mechanisms to
    aggregate data across trials from one bach recording.

    The assumption is that the provided POwlFile contains processed data
    including especially the "spiketrains".

    See also
    --------
    AdhocRecording : Calculates spiketains dynamically from unfiltered traces.
    """

    def __init__(self, powlfile: POwlFile):
        self.powlfile = powlfile
        self._cache: dict[str, Any] = {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(POwlFile({self.powlfile.filepath!r}))"

    def __str__(self) -> str:
        return f"{self!r}"

    def global_parameters(self) -> GroupParams:
        cache_key = "global_parameters"
        if cache_key not in self._cache:
            with self.powlfile as file:
                self._cache[cache_key] = get_params(file)
        return self._cache[cache_key]

    @cached_property
    def powl_version(self) -> str:
        if "powl_version" in self.global_parameters():
            return cast(str, self.global_parameters()["powl_version"])
        return "0.0.0"

    def get_samplingrate(self, which: str) -> float:
        if self.powl_version.startswith("0"):
            if which in self.global_parameters():
                return cast(float, self.global_parameters()[which])
            elif which in ("traces",):
                return cast(float, self.global_parameters()["adc_samplingrate"])
            elif which in ("stimuli",):
                return cast(float, self.global_parameters()["dac_samplingrate"])
        else:
            samplingrates: dict[str, float] = self.global_parameters()["samplingrates"]  # type: ignore
            if which in samplingrates:
                return cast(float, samplingrates[which])
            elif which in ("adc_samplingrate",):
                return cast(float, samplingrates["traces"])
            elif which in ("dac_samplingrate", "stimuli"):
                return cast(float, samplingrates["stimuli_freefield"])
        raise ValueError(f"Cannot find samplingrate for {which=}")

    @cached_property
    def session_id(self):
        global_params: dict = self.global_parameters()
        if "session" in global_params:
            owl = global_params["session"].get("owl", "unknown")
            date = global_params["session"].get("date", "unknown").replace("-", "")
            return f"{owl}_{date}"
        return f"unknown_unknown"

    def trials_parameters(self) -> dict[int, GroupParams]:
        cache_key = "trials_parameters"
        if cache_key not in self._cache:
            self._cache[cache_key] = {
                trial_index: get_params(trial_group)
                for trial_index, trial_group in self.powlfile.trials()
            }
        return self._cache[cache_key]

    def aggregate_trial_params(self, func: TrialParamFunc) -> list[Any]:
        params = self.trials_parameters()
        return [
            func(params[trial_index]) for trial_index in self.powlfile.trial_indexes
        ]

    def stimuli_parameters(self, stimulus_index=0) -> dict[int, GroupParams]:
        cache_key = f"stimulus_parameters({stimulus_index})"
        if cache_key not in self._cache:
            try:
                self._cache[cache_key] = {
                    trial_index: get_params(trial_group[f"stimuli/{stimulus_index}"])  # type: ignore
                    for trial_index, trial_group in self.powlfile.trials()
                }
            except KeyError:
                raise StimulusNotFound(recording=self, stimulus_index=stimulus_index)
        return self._cache[cache_key]

    def channel_numbers(self) -> list[int]:
        return self.powlfile.channel_numbers("spiketrains")

    def aggregate_stim_params(
        self,
        func: StimParamFunc,
        stimulus_index: int = 0,
    ) -> list[Any]:
        params = self.stimuli_parameters(stimulus_index)
        return [
            func(params[trial_index]) for trial_index in self.powlfile.trial_indexes
        ]

    def spike_trains(self, channel_number: int) -> SpikeTrainsType:
        cache_key = f"spike_trains({channel_number})"
        if cache_key not in self._cache:
            spiketrain_dict: SpikeTrainsType = {}
            data_key = f"spiketrains/{channel_number}/data_array"
            for trial_index, trial_group in self.powlfile.trials():
                spiketrain_dict[trial_index] = np.asarray(trial_group[data_key])
            self._cache[cache_key] = spiketrain_dict
        return self._cache[cache_key]

    def aggregrate_spikes(
        self,
        func: SpikeTimesFunc,
        *arg_lists: Unpack[tuple[Iterable[Any], ...]],
        channel_number: int,
    ) -> list[Any]:
        spiketrains = self.spike_trains(channel_number=channel_number)
        return [
            func(spiketrains[trial_index], *args)
            for trial_index, *args in zip(self.powlfile.trial_indexes, *arg_lists)
        ]

    @staticmethod
    def _agg_response_rates(st: SpikeTimesType, delay: float, dur: float) -> int:
        return np.searchsorted(st, delay + dur) - np.searchsorted(st, delay)  # type: ignore

    def response_rates(self, channel_number: int, stimulus_index: int) -> list[int]:
        return self.aggregrate_spikes(
            self._agg_response_rates,
            self.aggregate_stim_params(stim_delay, stimulus_index=stimulus_index),
            self.aggregate_stim_params(stim_len, stimulus_index=stimulus_index),
            channel_number=channel_number,
        )

    def stim_spiketrains(
        self, channel_number: int, stimulus_index: int = 0, ignore_onset: float = 0.0
    ) -> list[SpikeTimesType]:
        _agg_spike_times = lambda st, delay, dur: st[
            np.searchsorted(st, delay + ignore_onset) : np.searchsorted(st, delay + dur)
        ]
        return self.aggregrate_spikes(
            _agg_spike_times,
            self.aggregate_stim_params(stim_delay, stimulus_index=stimulus_index),
            self.aggregate_stim_params(stim_len, stimulus_index=stimulus_index),
            channel_number=channel_number,
        )

    def lfp_snippets(self, channel_number: int) -> LFPSnippetsType:
        cache_key = f"lfp_snippets({channel_number})"
        if cache_key not in self._cache:
            lfp_dict: LFPSnippetsType = {}
            data_key = f"lfp/{channel_number}/data_array"
            for trial_index, trial_group in self.powlfile.trials():
                lfp_dict[trial_index] = np.asarray(trial_group[data_key])
            self._cache[cache_key] = lfp_dict
        return self._cache[cache_key]

    def aggregrate_lfps(
        self, func: LFPSnippetFunc, *arg_lists: tuple[Iterable], channel_number: int
    ) -> list[Any]:
        lfpsnippets = self.lfp_snippets(channel_number=channel_number)
        return [
            func(lfpsnippets[trial_index], *args)
            for trial_index, *args in zip(self.powlfile.trial_indexes, *arg_lists)
        ]

    @cached_property
    def fileinfo(self):
        m = re.match(
            r"(?P<index>[0-9]+)_(?P<paradigm>.*?)(?:_(?P<number>[0-9]+))?\.h5",
            self.filename,
        )
        if m is None:
            return None
        return m.groupdict()

    @cached_property
    def filename(self):
        return os.path.basename(self.powlfile.filepath)
