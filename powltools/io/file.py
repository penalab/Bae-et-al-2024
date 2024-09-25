"""Convenience functions to work with pOwl data files"""

from __future__ import annotations
from functools import cached_property
import shutil
from typing import Any, Iterator
import h5py
import numpy.typing as npt


class POwlFile:
    KEY_TRIALS = "trials"
    KEY_PAUSES = "pauses"

    def __init__(self, filepath: str, *, mode: str = "r") -> None:
        self.filepath = filepath
        self.mode = mode
        self._handle_count = 0

    def __repr__(self) -> str:
        return f"POwlFile({self.filepath!r}, mode={self.mode!r})"

    def __enter__(self):
        if self._handle_count == 0:
            self._file = h5py.File(self.filepath, mode=self.mode)
        self._handle_count += 1
        return self._file

    def __exit__(self, exc_type, exc_value, traceback):
        self._handle_count -= 1
        if self._handle_count == 0:
            self._file.close()
            del self._file

    def trials(self) -> Iterator[tuple[int, h5py.Group]]:
        """Iterator over trials, yields tuples of (trial_index, trial_group)

        This is meant to be used similar to dict.items()

        trial_index is a int
        trial_group is an h5py.Group object

        Note that the file might be closed once this iterator is exhausted,
        to avoid this, you can place any loop within another with statement
        context.
        """
        with self as file:
            trials_group: h5py.Group = file.get(self.KEY_TRIALS)
            for trial_index in self.trial_indexes:
                yield trial_index, trials_group[f"{trial_index}"]

    @cached_property
    def trial_indexes(self) -> list[int]:
        with self as file:
            return sorted(int(k) for k in file[self.KEY_TRIALS].keys())

    @cached_property
    def has_pauses(self) -> bool:
        with self as file:
            return self.KEY_PAUSES in file

    def get_pause(self, pause_index: int) -> h5py.Group:
        if not self.has_pauses:
            raise PauseNotFound(pause_index, self)
        with self as file:
            try:
                return file[self.KEY_PAUSES][f"{pause_index}"]
            except KeyError:
                raise PauseNotFound(pause_index, self)

    def channel_numbers(
        self,
        signal_key: str = "traces",
        trial_index: int = 0,
    ) -> list[int]:
        with self as file:
            group: h5py.Group = file[f"{self.KEY_TRIALS}/{trial_index}/{signal_key}"]
            return sorted(int(k) for k in group.keys())

    def write_data(
        self,
        data_dict: dict[int, npt.NDArray],
        channel_number: int,
        signal_key: str,
    ) -> None:
        group_key = f"{signal_key}/{channel_number}"
        for trial_index, trial_group in self.trials():
            data_group = trial_group.create_group(group_key)
            data_group.create_dataset(
                name="data_array",
                data=data_dict[trial_index],
            )

    def remove_signal(self, signal_key: str):
        for _, trial_group in self.trials():
            del trial_group[signal_key]

    def consolidate(self):
        if not self.mode in ("r+",):
            raise ValueError("POwlFile must be in mode='r+' to be consolidated.")
        if self._handle_count > 0:
            raise IOError("POwlFile cannot be consolidated while open.")
        temp_filename = self.filepath + ".tmp"
        with self as file_orig:
            with h5py.File(temp_filename, mode="x") as file_new:
                for k in file_orig.keys():
                    file_orig.copy(k, file_new)
        shutil.move(temp_filename, self.filepath)


class PauseNotFound(KeyError):
    def __init__(
        self,
        pause_index: Any,
        powlfile: POwlFile,
    ) -> None:
        self.pause_index = pause_index
        self.powlfile = powlfile
        super().__init__()

    def __str__(self) -> str:
        with self.powlfile as file:
            if not self.powlfile.has_pauses:
                return f"{self.powlfile!r} has no pauses"
            if not self.pause_index in file[self.powlfile.KEY_PAUSES]:
                return f"{self.powlfile!r} has no pause with index {self.pause_index}"
