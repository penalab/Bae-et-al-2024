from __future__ import annotations as _annotations
from functools import cached_property

import os
import json
import re
import numpy as np
from typing import Callable, Optional, cast
from ..io.file import POwlFile
from ..io.analog import AnalogSignal, SignalType
from ..filters.chained import get_spikecontinuous_filter
from ..spikes.detect import detect_spikes, spiketrains2times, split_indexes
from .recording import Recording, SpikeTrainsType


class AdhocRecording(Recording):
    """Extension of Recording that can calculate spiketrains on the fly

    Other than the parent class, files not expected to contain "spiketrains",
    but are expected to still contain the original wideband recorded signal
    under "traces".

    The AdhocRecording instance has an attribute `thresholds`, a dictionary
    that maps channel numbers (int) to thresholds (float) in microvolts to be
    compliant with the thresholding tool and data saved in `annotations.json`.

    Thresholds can be manipulated in various ways:

    *   You can manually set threshold

    """

    DEFAULT_THRESHOLD_SD_FACTOR = -3.5

    def default_threshold_func(
        self,
        channel_number: int,
        spikecontinuous_signal: SignalType,
    ) -> float:
        return np.std(spikecontinuous_signal).item() * self.DEFAULT_THRESHOLD_SD_FACTOR

    def __init__(
        self,
        powlfile: POwlFile,
        *,
        treshold_func: Optional[Callable[[int, SignalType], float]] = None,
    ):
        super().__init__(powlfile)
        self._thresholds = {}
        self.threshold_func = (
            treshold_func if not treshold_func is None else self.default_threshold_func
        )

    @cached_property
    def powl_version(self):
        return self.global_parameters().get("powl_version", "0.0.0")

    @cached_property
    def samplingrate(self) -> float:
        if self.powl_version == "0.0.0":
            return cast(float, self.global_parameters()["adc_samplingrate"])
        else:
            return cast(float, self.global_parameters()["samplingrates"]["traces"])  # type: ignore

    @cached_property
    def session_id(self):
        dirname = os.path.basename(os.path.dirname(self.powlfile.filepath))
        if m := re.match(
            "(?P<date>[0-9]{8})_(?P<owl>[^_]+)_(?P<condition>.*)$", dirname
        ):
            owl = m.group("owl")
            date = m.group("date")
            return f"{owl}_{date}"
        return f"unknown_preliminary"

    def channel_numbers(self):
        return self.powlfile.channel_numbers("traces")

    def _threshold_from_file(self, channel_number: int) -> float:
        powlfile_path = self.powlfile.filepath
        dirname = os.path.dirname(powlfile_path)
        annotations_path = os.path.join(dirname, "annotations.json")
        if not os.path.isfile(annotations_path):
            raise FileNotFoundError("No annotations file found")
        with open(annotations_path, "r") as annotations_file:
            annotations = json.load(annotations_file)
        old_threshold = annotations["files"][os.path.basename(powlfile_path)][
            "thresholds"
        ][f"{channel_number}"]
        return float(old_threshold.removesuffix("e-6"))

    def get_threshold(self, channel_number: int):
        if channel_number not in self._thresholds:
            try:
                threshold = self._threshold_from_file(channel_number)
            except:
                anas = self.spike_continuous(channel_number)
                threshold = 1e6 * self.threshold_func(
                    channel_number, anas.continuous_signal
                )
            finally:
                self.set_threshold(channel_number, threshold)
        return self._thresholds[channel_number]

    def set_threshold(self, channel_number: int, value: float) -> None:
        self._thresholds[channel_number] = value
        # Clear spike_trains cache:
        self._cache.pop(f"spike_trains({channel_number})", None)

    def spike_trains(self, channel_number: int) -> SpikeTrainsType:
        cache_key = f"spike_trains({channel_number})"
        if cache_key not in self._cache:
            anas = self.spike_continuous(channel_number=channel_number)
            current_threshold = self.get_threshold(channel_number)
            self._cache[cache_key] = spiketrains2times(
                split_indexes(
                    detect_spikes(anas.continuous_signal, 1e-6 * current_threshold),
                    anas.trials_start_stop,
                ),
                self.samplingrate,
            )
        return self._cache[cache_key]

    def spike_continuous(self, channel_number: int) -> AnalogSignal:
        cache_key = f"spike_continuous({channel_number})"
        if cache_key not in self._cache:
            analog_unfiltered = AnalogSignal(
                self.powlfile, channel_number=channel_number, signal_key="traces"
            )
            analog_unfiltered.load()
            self._cache[cache_key] = analog_unfiltered.copy_filtered(
                filter_func=get_spikecontinuous_filter(fs=self.samplingrate),
                signal_key="spikecontinuous",
            )
        return self._cache[cache_key]
