"""Functions to detect and organize spikes"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt

SignalType = npt.NDArray[np.float64]
SpikeIndexesType = npt.NDArray[np.int_]
SpikeTrainsType = dict[int, SpikeIndexesType]
SpikeTimesType = npt.NDArray[np.float64]
SpikeTrainsTimeType = dict[int, SpikeTimesType]
StartStopDict = dict[int, tuple[int, int]]


def detect_spikes(
    analog_signal: SignalType,
    threshold: float,
) -> SpikeIndexesType:
    beyond_thresh = (
        (analog_signal < threshold) if threshold < 0 else (analog_signal > threshold)
    )
    indexes = np.where(np.diff(beyond_thresh.astype(int)) > 0)[0] + 1
    return indexes


def split_indexes(
    indexes: SpikeIndexesType,
    start_stop_dict: StartStopDict,
) -> SpikeTrainsType:
    spiketrains_dict: SpikeTrainsType = {}
    for key, (start, stop) in start_stop_dict.items():
        indexes_start = np.searchsorted(indexes, start)
        indexes_stop = np.searchsorted(indexes, stop)
        spiketrains_dict[key] = indexes[indexes_start:indexes_stop] - start
    return spiketrains_dict


def unsplit_indexes(
    spiketrains_dict: SpikeTrainsType,
    start_stop_dict: StartStopDict,
) -> SpikeIndexesType:
    indexes_list: list[SpikeIndexesType] = []
    for key, (start, stop) in start_stop_dict.items():
        indexes_list.append(spiketrains_dict[key] + start)
    return np.hstack(indexes_list)


def indexes2times(indexes: SpikeIndexesType, fs: float) -> SpikeTimesType:
    return indexes / fs


def spiketrains2times(
    spiketrains_dict: SpikeTrainsType, fs: float
) -> SpikeTrainsTimeType:
    spiketrains: SpikeTrainsTimeType = {
        key: indexes2times(spiketrain, fs)
        for key, spiketrain in spiketrains_dict.items()
    }
    return spiketrains
