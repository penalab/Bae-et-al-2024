from __future__ import annotations
import numpy as np
import numpy.typing as npt

SignalType = npt.NDArray[np.float64]
SpikeIndexesType = npt.NDArray[np.int_]
SpikeTrainsType = dict[int, SpikeIndexesType]
SpikeTimesType = npt.NDArray[np.float64]
SpikeTrainsTimeType = dict[int, SpikeTimesType]
StartStopDict = dict[int, tuple[int, int]]
WaveformsType = npt.NDArray[np.float64]
WaveformsDictType = dict[int, WaveformsType]


def extract_waveforms(
    spikeindexes: SpikeIndexesType,
    analog_signal: SignalType,
    leading: int = 15,
    trailing: int = 25,
    fill_value: float = np.nan,
) -> WaveformsType:
    waveforms = np.empty(
        (spikeindexes.size, leading + trailing),
        dtype=analog_signal.dtype,
    )
    right = analog_signal.size - trailing
    for i, index in enumerate(spikeindexes):
        if index < leading:
            # left-incomplete waveform
            waveforms[i, : leading - index].fill(fill_value)
            waveforms[i, leading - index :] = analog_signal[: index + trailing]
        elif index > right:
            # right-incomplete waveform
            waveforms[i, right - index :].fill(fill_value)
            waveforms[i, : right - index] = analog_signal[index - leading :]
        else:
            waveforms[i, :] = analog_signal[index - leading : index + trailing]
    return waveforms


def split_indexes_and_waveforms(
    indexes: SpikeIndexesType,
    waveforms: WaveformsType,
    start_stop_dict: StartStopDict,
) -> tuple[SpikeTrainsType, dict[int, WaveformsType]]:
    spiketrains_dict: SpikeTrainsType = {}
    waveforms_dict: dict[int, WaveformsType] = {}
    for key, (start, stop) in start_stop_dict.items():
        indexes_start = np.searchsorted(indexes, start)
        indexes_stop = np.searchsorted(indexes, stop)
        spiketrains_dict[key] = indexes[indexes_start:indexes_stop] - start
        waveforms_dict[key] = waveforms[indexes_start:indexes_stop, :]
    return spiketrains_dict, waveforms_dict
