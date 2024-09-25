"""processing.py

Usage: python -m processing [--copy] [--verbose] <annotations-path>

<annotations-path> is the path to the annotations.json file of the session to
be processed. Multiple paths are accepted as separate arguments or glob pattern(s).

Options:

--help      Print this help and exit
--copy      Copy each .h5 file before processing to original-filename_copy.h5
            Existing copies will be overwritten without warning.
            Default is to overwrite the original files.
--verbose   Show more output (console logger level to DEBUG)

"""

from __future__ import annotations as _annotations
import datetime
from functools import partial
import logging
import os
import shutil
import json
import time
from powltools.analysis.recording import Recording
from powltools.io.file import POwlFile
from powltools.io.analog import AnalogSignal, AnalogSnippets
from powltools.filters.chained import get_spikecontinuous_filter
from powltools.io.parameters import get_params, save_params
from powltools.lfp.generate import filter_lfp
from powltools.spikes.detect import detect_spikes, spiketrains2times
from powltools.spikes.waveforms import extract_waveforms, split_indexes_and_waveforms

logger = logging.getLogger("powl")
logger.setLevel(logging.DEBUG)
log2console = logging.StreamHandler()
log2console.setLevel(logging.INFO)
log2console.setFormatter(
    logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
)
logger.addHandler(log2console)


def copy_file(filepath: str) -> str:
    filepath_copy = "{}_copy{}".format(*os.path.splitext(filepath))
    filename = os.path.basename(filepath_copy)
    logger.info(f"Copy file to {filename}")
    shutil.copy2(filepath, filepath_copy)
    logger.info("File copied")
    return filepath_copy


def process_channel(pf: POwlFile, channel_number: int, threshold: float):
    with pf as file:
        global_params = get_params(file)
    samplingrate = Recording(pf).get_samplingrate("traces")
    spikecontinuous_filter = get_spikecontinuous_filter(fs=samplingrate)
    logger.debug("Create and load AnalogSignal wideband from traces")
    wideband = AnalogSignal(pf, channel_number=channel_number, signal_key="traces")
    wideband.load()
    logger.debug("Copy-filter wideband to spikecontinous")
    anas_spikecont = wideband.copy_filtered(
        filter_func=spikecontinuous_filter,
        signal_key="spikecontinuous",
        astype=wideband.continuous_signal.dtype,
    )
    # Would save spikecontinuous into file:
    # anas_spikecont.save()
    logger.debug("Detect spikes")
    spikesindexes = detect_spikes(anas_spikecont.continuous_signal, threshold=threshold)
    logger.debug("Extract waveforms")
    waveforms = extract_waveforms(
        spikesindexes,
        anas_spikecont.continuous_signal,
        leading=15,
        trailing=25,
    )
    logger.debug("split spikeindexes and waveforms")
    spikeindexes_by_trial, waveforms_by_trial = split_indexes_and_waveforms(
        indexes=spikesindexes,
        waveforms=waveforms,
        start_stop_dict=anas_spikecont.trials_start_stop,
    )
    logger.debug("convert spiketrains from indexes to seconds")
    spiketimes_by_trial = spiketrains2times(spikeindexes_by_trial, fs=samplingrate)
    logger.debug("Write spiketimes to file as spiketrains")
    pf.write_data(
        data_dict=spiketimes_by_trial,
        channel_number=channel_number,
        signal_key="spiketrains",
    )
    logger.debug("Write waveforms to file as waveforms")
    pf.write_data(
        data_dict=waveforms_by_trial,
        channel_number=channel_number,
        signal_key="waveforms",
    )
    simple_filter_lfp = partial(
        filter_lfp,
        spikeindexes=spikesindexes,
        samplingrate=samplingrate,
    )
    logger.debug("Copy-filter wideband to lfp_continuous")
    lfp_continuous = wideband.copy_filtered(
        filter_func=simple_filter_lfp,
        signal_key="lfp_continuous",
    )
    logger.debug("Resample lfp_continuous to AnalogSnippets lfp")
    lfp = AnalogSnippets.resampled_from_analog_signal(
        lfp_continuous,
        signal_key="lfp",
        up=1024,
        down=25000,
        pauses=False,
        astype=wideband.continuous_signal.dtype,
    )
    logger.debug("Write lfp snippets to file")
    lfp.save()


def process_session(annotations_file: str, copy_first=True):
    with open(annotations_file, "r") as f:
        annotations = json.load(f)
    dirname = os.path.dirname(annotations_file)

    for filename, fileinfo in annotations["files"].items():
        logger.info(f"Processing {filename}")
        filepath = os.path.join(dirname, filename)
        if copy_first:
            filepath = copy_file(filepath)
            filename = os.path.basename(filepath)

        pf = POwlFile(filepath, mode="r+")
        rec = Recording(pf)
        for channel_number in pf.channel_numbers():
            chan_str = str(channel_number)
            if chan_str in annotations.get("drop_channels", {}):
                logger.info(f"Drop channel {channel_number}")
                continue
            logger.info(f"Processing channel {channel_number} of {filename}")
            if chan_str in fileinfo["thresholds"]:
                threshold = float(fileinfo["thresholds"][chan_str])
            else:
                threshold = -32.0e-6
                logger.warning(f"Using default {threshold = }")
            process_channel(pf, channel_number=channel_number, threshold=threshold)
        with pf as file:
            global_params = rec.global_parameters().copy()
            global_params["session"] = annotations["session"]
            global_params["regions"] = {
                channel_number: region
                for channel_number, region in annotations["regions"].items()
                if not channel_number in annotations.get("drop_channels", {})
            }
            global_params["signals"] = {
                "lfp": {"samplingrate": 1000.0},
                "stimuli": {"samplingrate": rec.get_samplingrate("stimuli")},
                "waveforms": {
                    "samplingrate": rec.get_samplingrate("traces"),
                    "leading": 15,
                    "trailing": 25,
                },
            }
            logger.debug("Removing old global parameters")
            del file["parameters"]
            logger.debug("Writing new global parameters")
            save_params(global_params, file)

        logger.info("Remove (wideband) traces:")
        pf.remove_signal(signal_key="traces")
        logger.info("Copy fresh file without trash:")
        pf.consolidate()
        logger.info(f"Finished Processing {filename}")


def processing_help():
    print(__doc__)


def main(annotation_filenames: list[str], copy_first=False):
    logfile_date_str = f"{datetime.datetime.now():%Y%m%d-%H%M}"
    log2file_handlers = {}
    for dirname in set(
        [os.path.dirname(filename) for filename in annotation_filenames]
    ):
        log2file = logging.FileHandler(
            os.path.join(dirname, f"{logfile_date_str}_powl_processing_debug.log")
        )
        log2file.setLevel(logging.DEBUG)
        log2file.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        log2file_handlers[dirname] = log2file

    for annotations_file in annotation_filenames:
        logger.addHandler(log2file_handlers[os.path.dirname(annotations_file)])
        process_session(annotations_file, copy_first=copy_first)
        logger.removeHandler(log2file_handlers[os.path.dirname(annotations_file)])
    return 0


if __name__ == "__main__":
    import sys
    import os
    import glob

    args = sys.argv[1:]

    if "--help" in args:
        processing_help()
        raise SystemExit()

    if "--verbose" in args:
        log2console.setLevel(logging.DEBUG)
        args.remove("--verbose")

    if "--copy" in args:
        copy_first = True
        args.remove("--copy")
    else:
        copy_first = False

    annotation_filenames = sorted(
        set([os.path.abspath(p) for arg in sys.argv[1:] for p in glob.glob(arg)])
    )
    if not annotation_filenames:
        logger.error("No annotation files were found")
        raise SystemExit(1)

    raise SystemExit(main(annotation_filenames, copy_first=copy_first))
