from itertools import combinations, repeat
import pathlib

import numpy as np
import numpy.typing as npt
import scipy.stats
import scipy.signal
import pandas as pd
from powltools.io.file import POwlFile
from powltools.analysis.recording import Recording
from powltools.analysis.recording import stim_level
from powltools.analysis.recording import stim_delay
from powltools.analysis.recording import stim_len
from powltools.analysis.recording import stim_f_am
from powltools.analysis.recording import group_by_param
from powltools.filters.offlinefilters import bandpass_filter

from xcorr_tools import binary_spiketrain
from xcorr_tools import make_psth_bins
from xcorr_tools import stim_psth_bins
from xcorr_tools import base_psth_bins
from xcorr_tools import cross_correlation
from xcorr_tools import get_peak
from xcorr_tools import fixed_var_stimuli
from xcorr_tools import iter_session_values
from xcorr_tools import filter_rate_level_am
from xcorr_tools import filter_rate_level_out_am
from xcorr_tools import filter_relative_level_am
from xcorr_tools import get_power
from xcorr_tools import get_phaselocking
from xcorr_tools import get_phaselocking_am_stimuli


def main():
    # Single stimulus data, flat noise, curated/clean
    # Contains one line per [date, owl, channel, level]
    OUTDIR = pathlib.Path("./forebrain_intermediate_results").absolute()
    OUTDIR.mkdir(exist_ok=True)

    data_dir = r"E:\Andrea-Freefield"
    df = pd.read_csv(r"./pooled_data_excel/auditory_units_combined_am.csv")
    df.set_index(["date", "owl"], inplace=True)
    df.sort_index(inplace=True)
    region = "dualregion"

    # Single stim:
    print("singlestim_ccg".upper())
    single_ccg_df = singlestim_ccg(df, data_dir)
    single_ccg_df.to_feather(OUTDIR / f"am_single_ccg_{region}.feather")
    print("singlestim_rlf".upper())
    singlestim_rlf_df = singlestim_rlf(df, data_dir)
    print(singlestim_rlf_df)
    singlestim_rlf_df.to_feather(OUTDIR / f"am_single_rlf_{region}.feather")

    print("singlestim_rlf_out".upper())
    singlestim_rlf_out_df = singlestim_rlf_out(df, data_dir)
    #        print(singlestim_rlf_df)
    singlestim_rlf_out_df.to_feather(OUTDIR / f"am_single_rlf_out_{region}.feather")

    print("singlestim_gamma_power".upper())
    singlestim_gamma_power_df = singlestim_gamma_power(df, data_dir)
    singlestim_gamma_power_df.to_feather(
        OUTDIR / f"am_single_gamma_power_{region}.feather"
    )
    print("singlestim_stim_phaselocking".upper())
    singlestim_phaselocking_df = singlestim_phaselocking(df, data_dir)
    singlestim_phaselocking_df.to_feather(
        OUTDIR / f"am_single_stim_phaselocking_{region}.feather"
    )

    # Two Stim:
    print("twostim_ccg".upper())
    twostim_ccg_df = twostim_ccg(df, data_dir)
    twostim_ccg_df.to_feather(OUTDIR / f"am_twostim_ccg_{region}.feather")
    print("twostim_rlf".upper())
    twostim_rlf_df = twostim_rlf(df, data_dir)
    twostim_rlf_df.to_feather(OUTDIR / f"am_twostim_rlf_{region}.feather")
    print("twostim_gamma_power".upper())
    twostim_gamma_power_df = twostim_gamma_power(df, data_dir)
    twostim_gamma_power_df.to_feather(
        OUTDIR / f"am_twostim_gamma_power_{region}.feather"
    )
    print("twostim_stim_phaselocking".upper())
    twostim_phaselocking_df = twostim_phaselocking(df, data_dir)
    twostim_phaselocking_df.to_feather(
        OUTDIR / f"am_twostim_stim_phaselocking_{region}.feather"
    )

    return 0


struggled_during_am = {
    55: [
        ("2023-05-07", "33"),
        ("2023-05-08", "33"),
        ("2023-05-10", "33"),
        ("2023-05-11", "33"),
        ("2023-06-05", "33"),
    ],
    75: [
        ("2023-04-17", "33"),
        ("2023-04-20", "33"),
        ("2023-04-28", "33"),
        ("2023-05-08", "33"),
        ("2023-05-17", "33"),
    ],
}


def get_latency(trace, time_bins):
    max_ind = np.argmax(trace)
    half_ind = np.argmax(trace >= trace[max_ind] / 2)
    latency = time_bins[half_ind]
    peak_time = time_bins[max_ind]
    return latency, peak_time


def singlestim_ccg(df, data_dir):
    singlestim_ccg = []
    for session in iter_session_values(
        df, filename_filter=filter_rate_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            modulation_frequencies = set(rec.aggregate_stim_params(stim_f_am))
            modulation_frequency = modulation_frequencies.pop()
            if modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del modulation_frequencies

            if (session["date"], session["owl"]) in struggled_during_am[
                modulation_frequency
            ]:
                continue

            psth_bins = rec.aggregate_stim_params(stim_psth_bins)
            base_bins = rec.aggregate_stim_params(base_psth_bins)

            varying_levels = np.array(
                rec.aggregate_stim_params(stim_level, stimulus_index=0)
            )
            unique_levels = np.unique(varying_levels)
            fixed_azimuths = np.array(
                rec.aggregate_stim_params(lambda params: params["azi"])
            )
            fixed_elevations = np.array(
                rec.aggregate_stim_params(lambda params: params["ele"])
            )

            # Binary spiketrains during and before stimuli for all channels
            stim_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, psth_bins, channel_number=chan
                    )
                )
                for chan in channels
            }
            base_spiketrains = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, base_bins, channel_number=chan
                    )
                )
                for chan in channels
            }

            for i, (chan1, chan2) in enumerate(combinations(channels, 2)):
                region_1 = regions[str(chan1)]
                region_2 = regions[str(chan2)]

                if region_1 == region_2:
                    if (region_1 == "OT") & (region_2 == "OT"):
                        corr_type = "within_OT"
                    elif (region_1 == "Forebrain") & (region_2 == "Forebrain"):
                        corr_type = "within_Forebrain"
                else:
                    corr_type = "cross_region"

                if chan1 < 17:
                    depth_order = [
                        14,
                        11,
                        3,
                        6,
                        2,
                        7,
                        1,
                        8,
                        4,
                        5,
                        16,
                        9,
                        15,
                        10,
                        13,
                        12,
                    ]
                elif chan1 >= 17:
                    depth_order = [
                        30,
                        27,
                        19,
                        22,
                        18,
                        23,
                        17,
                        24,
                        20,
                        21,
                        32,
                        25,
                        31,
                        26,
                        29,
                        28,
                    ]

                chan1_index = depth_order.index(chan1)

                if chan2 < 17:
                    depth_order = [
                        14,
                        11,
                        3,
                        6,
                        2,
                        7,
                        1,
                        8,
                        4,
                        5,
                        16,
                        9,
                        15,
                        10,
                        13,
                        12,
                    ]
                elif chan2 >= 17:
                    depth_order = [
                        30,
                        27,
                        19,
                        22,
                        18,
                        23,
                        17,
                        24,
                        20,
                        21,
                        32,
                        25,
                        31,
                        26,
                        29,
                        28,
                    ]

                chan2_index = depth_order.index(chan2)

                depth_distance = abs(chan1_index - chan2_index)

                for level in unique_levels:
                    if not level in (-5, -20):
                        # We only need the levels that were used for drivers in competition
                        continue

                    # Boolean array to select trials of this level:
                    mask = varying_levels == level
                    # Data for this condition:
                    psth_u1 = stim_spiketrains[chan1][mask]
                    psth_u2 = stim_spiketrains[chan2][mask]
                    base_u1 = base_spiketrains[chan1][mask]
                    base_u2 = base_spiketrains[chan2][mask]

                    lags = (
                        scipy.signal.correlation_lags(
                            psth_u1.shape[1], psth_u2.shape[1]
                        )
                        * 0.001
                    )

                    # Mean firing rates:
                    resp_rate_u1: float = psth_u1.sum() / psth_u1.shape[0]
                    resp_rate_u2: float = psth_u2.sum() / psth_u2.shape[0]
                    base_rate_u1: float = base_u1.sum() / base_u1.shape[0]
                    base_rate_u2: float = base_u2.sum() / base_u2.shape[0]
                    # Geometric means:
                    gm_resp_rate = (resp_rate_u1 * resp_rate_u2) ** 0.5
                    gm_base_rate = (base_rate_u1 * base_rate_u2) ** 0.5
                    # Exclude of levels with no response
                    if gm_resp_rate <= gm_base_rate:
                        continue

                    ccg, shuff_ccg = cross_correlation(psth_u1, psth_u2, lags)

                    # Normalize by (geometric) mean response rate and psth length
                    norm_ccg = (
                        (ccg - np.mean(ccg)) / (gm_resp_rate) / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])
                    norm_shuff = (
                        (shuff_ccg - np.mean(shuff_ccg))
                        / (gm_resp_rate)
                        / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])

                    smscorrected = norm_ccg - norm_shuff

                    ccg_peak = get_peak(
                        smscorrected, lags, lag_window=0.015, baseline_window=0.05
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.05
                    )
                    ccg_peak_uncorrected = get_peak(
                        norm_ccg, lags, lag_window=0.015, baseline_window=0.05
                    )

                    if ccg_peak["peak_corr"] > 5 * ccg_peak["baseline_std"]:
                        tmp = {
                            "date": session["date"],
                            "owl": session["owl"],
                            "channel1": chan1,
                            "channel2": chan2,
                            "azimuth": fixed_azimuths[0],
                            "elevation": fixed_elevations[0],
                            "intensity": level,
                            "modulation_frequency": modulation_frequency,
                            "xcorr_peak": ccg_peak["peak_corr"],
                            "peak_time": ccg_peak["peak_lag"],
                            "synchrony_val": ccg_peak["peak_area"],
                            "xcorr_width": ccg_peak["peak_width"],
                            "xcorr_peak_shuff": ccg_peak_shuff["peak_corr"],
                            "peak_time_shuff": ccg_peak_shuff["peak_lag"],
                            "synchrony_val_shuff": ccg_peak_shuff["peak_area"],
                            "xcorr_width_shuff": ccg_peak_shuff["peak_width"],
                            "stimlocked_peak": ccg_peak_uncorrected["peak_corr"],
                            "stim_locked_peak_time": ccg_peak_uncorrected["peak_lag"],
                            "stimlocked_synchrony_val": ccg_peak_uncorrected[
                                "peak_area"
                            ],
                            "hemisphere": hemisphere,
                            "stimtype": "single",
                            "ccg": smscorrected,
                            "ccg_shuff": norm_shuff,
                            "ccg_uncorrected": norm_ccg,
                            "corr_type": corr_type,
                            "depth_distance": depth_distance,
                        }
                        singlestim_ccg.append(tmp)

    single_ccg_df = pd.DataFrame(singlestim_ccg)
    return single_ccg_df


def singlestim_rlf(df, data_dir):
    singlestim_rlf = []
    for session in iter_session_values(
        df, filename_filter=filter_rate_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            modulation_frequencies = set(rec.aggregate_stim_params(stim_f_am))
            modulation_frequency = modulation_frequencies.pop()
            if modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del modulation_frequencies

            if (session["date"], session["owl"]) in struggled_during_am[
                modulation_frequency
            ]:
                continue

            trial_delays = set(rec.aggregate_stim_params(stim_delay, stimulus_index=0))
            trial_durations = set(rec.aggregate_stim_params(stim_len, stimulus_index=0))
            delay = trial_delays.pop()
            duration = trial_durations.pop()
            if any([trial_delays, trial_durations]):
                raise ValueError(
                    "Stimulus delay or durcation not the same for all trials"
                )
            del trial_delays, trial_durations

            latency_bins = make_psth_bins(0, delay + duration, binsize=0.001, offset=0)

            trial_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain,
                        repeat(latency_bins),
                        channel_number=chan,
                    )
                )
                for chan in list(channels)
            }

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=0,
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=0,
                )
            )
            azimuth = fixed_azimuths.pop()
            elevation = fixed_elevations.pop()
            if any([fixed_azimuths, fixed_elevations]):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations

            for chan in channels:
                region_chan = regions[str(chan)]
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                trial_levels = np.array(
                    rec.aggregate_stim_params(stim_level, stimulus_index=0)
                )
                resp_by_level = group_by_param(resp, trial_levels)

                psth = trial_spiketrains[chan]
                psth_by_level = group_by_param(psth, trial_levels)

                for level, level_resp in resp_by_level.items():
                    mean_psth = np.mean(psth_by_level[level], axis=0)
                    time_firstspike, time_peak = get_latency(
                        mean_psth[latency_bins[:-1] >= delay],
                        latency_bins[latency_bins >= delay],
                    )
                    time_firstspike = time_firstspike - delay
                    time_peak = time_peak - delay
                    # if (time_firstspike < 0.0) or (time_peak < 0.0) or (time_firstspike > 0.040) or (time_peak > 0.040):
                    #     continue
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "intensity": level,
                        "modulation_frequency": modulation_frequency,
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "psth": mean_psth,
                        "first_spike_latency": time_firstspike,
                        "max_peak_latency": time_peak,
                        "hemisphere": hemisphere,
                        "stimtype": "singlestim",
                        "region": region_chan,
                    }
                    singlestim_rlf.append(tmp)
    singlestim_rlf_df = pd.DataFrame(singlestim_rlf)
    return singlestim_rlf_df


def singlestim_rlf_out(df, data_dir):
    singlestim_rlf = []
    for session in iter_session_values(
        df, filename_filter=filter_rate_level_out_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            modulation_frequencies = set(rec.aggregate_stim_params(stim_f_am))
            modulation_frequency = modulation_frequencies.pop()
            if modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del modulation_frequencies

            if (session["date"], session["owl"]) in struggled_during_am[
                modulation_frequency
            ]:
                continue

            trial_delays = set(rec.aggregate_stim_params(stim_delay, stimulus_index=0))
            trial_durations = set(rec.aggregate_stim_params(stim_len, stimulus_index=0))
            delay = trial_delays.pop()
            duration = trial_durations.pop()
            if any([trial_delays, trial_durations]):
                raise ValueError(
                    "Stimulus delay or durcation not the same for all trials"
                )
            del trial_delays, trial_durations

            latency_bins = make_psth_bins(0, delay + duration, binsize=0.001, offset=0)

            trial_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain,
                        repeat(latency_bins),
                        channel_number=chan,
                    )
                )
                for chan in list(channels)
            }

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=0,
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=0,
                )
            )
            azimuth = fixed_azimuths.pop()
            elevation = fixed_elevations.pop()
            if any([fixed_azimuths, fixed_elevations]):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations

            for chan in channels:
                region_chan = regions[str(chan)]
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                trial_levels = np.array(
                    rec.aggregate_stim_params(stim_level, stimulus_index=0)
                )
                resp_by_level = group_by_param(resp, trial_levels)

                psth = trial_spiketrains[chan]
                psth_by_level = group_by_param(psth, trial_levels)

                for level, level_resp in resp_by_level.items():
                    mean_psth = np.mean(psth_by_level[level], axis=0)
                    time_firstspike, time_peak = get_latency(
                        mean_psth[latency_bins[:-1] >= delay],
                        latency_bins[latency_bins >= delay],
                    )
                    time_firstspike = time_firstspike - delay
                    time_peak = time_peak - delay
                    # if (time_firstspike < 0.0) or (time_peak < 0.0) or (time_firstspike > 0.040) or (time_peak > 0.040):
                    #     continue
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "intensity": level,
                        "modulation_frequency": modulation_frequency,
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "psth": mean_psth,
                        "first_spike_latency": time_firstspike,
                        "max_peak_latency": time_peak,
                        "hemisphere": hemisphere,
                        "stimtype": "singlestim",
                        "region": region_chan,
                    }
                    singlestim_rlf.append(tmp)
    singlestim_rlf_df = pd.DataFrame(singlestim_rlf)
    return singlestim_rlf_df


def singlestim_gamma_power(df, data_dir):
    singlestim_gamma_power = []
    for session in iter_session_values(
        df, filename_filter=filter_rate_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            modulation_frequencies = set(rec.aggregate_stim_params(stim_f_am))
            modulation_frequency = modulation_frequencies.pop()
            if modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del modulation_frequencies

            if (session["date"], session["owl"]) in struggled_during_am[
                modulation_frequency
            ]:
                continue

            trial_levels = rec.aggregate_stim_params(stim_level, stimulus_index=0)
            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=0,
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=0,
                )
            )
            azimuth = fixed_azimuths.pop()
            elevation = fixed_elevations.pop()
            if any([fixed_azimuths, fixed_elevations]):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations

            for chan in channels:
                region_chan = regions[str(chan)]
                stim_spikes = np.array(
                    rec.stim_spiketrains(channel_number=chan, ignore_onset=0.050),
                    dtype="object",
                )
                spiketrains_by_level = group_by_param(stim_spikes, trial_levels)

                lfp_arr = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan)
                )
                lfp_arr = lfp_arr - np.mean(lfp_arr, axis=0)
                bandpass_lfp = bandpass_filter(
                    lfp_arr,
                    20,
                    50,
                    fs=lfp_samplingrate,
                    order=1,
                )

                trial_power = np.array(
                    [
                        get_power(
                            bandpass_lfp[trial_index],
                            baseline_start=trial_delays[trial_index]
                            - trial_durations[trial_index],
                            baseline_stop=trial_delays[trial_index] - 0.05,
                            stim_start=trial_delays[trial_index] + 0.05,
                            stim_stop=trial_delays[trial_index]
                            + trial_durations[trial_index],
                            samplingrate=lfp_samplingrate,
                        )
                        for trial_index in rec.powlfile.trial_indexes
                    ]
                )
                power_by_level = group_by_param(trial_power, trial_levels)
                lfp_by_level = group_by_param(bandpass_lfp, trial_levels)

                for level, level_power in power_by_level.items():
                    level_phaselocking = get_phaselocking(
                        lfp_by_level[level], spiketrains_by_level[level]
                    )
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "intensity": level,
                        "modulation_frequency": modulation_frequency,
                        "gammapower": np.mean(level_power),
                        "gammapower_sem": scipy.stats.sem(level_power),
                        "gamma_plv": level_phaselocking["vector_strength"],
                        "gamma_plv_angle": level_phaselocking["mean_phase"],
                        "gamma_plv_p": level_phaselocking["p"],
                        "hemisphere": hemisphere,
                        "stimtype": "singlestim",
                        "region": region_chan,
                    }
                    singlestim_gamma_power.append(tmp)
    singlestim_gamma_power_df = pd.DataFrame(singlestim_gamma_power)
    return singlestim_gamma_power_df


def singlestim_phaselocking(df, data_dir):
    singlestim_phaselocking = []

    for session in iter_session_values(
        df, filename_filter=filter_rate_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            modulation_frequencies = set(rec.aggregate_stim_params(stim_f_am))
            modulation_frequency = modulation_frequencies.pop()
            if modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del modulation_frequencies

            if (session["date"], session["owl"]) in struggled_during_am[
                modulation_frequency
            ]:
                continue

            trial_levels = rec.aggregate_stim_params(stim_level, stimulus_index=0)
            trial_delays = set(rec.aggregate_stim_params(stim_delay, stimulus_index=0))
            trial_durations = set(rec.aggregate_stim_params(stim_len, stimulus_index=0))
            delay = trial_delays.pop()
            duration = trial_durations.pop()
            if any([trial_delays, trial_durations]):
                raise ValueError(
                    "Stimulus delay or durcation not the same for all trials"
                )
            del trial_delays, trial_durations

            latency_bins = make_psth_bins(0, delay + duration, binsize=0.001, offset=0)

            trial_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain,
                        repeat(latency_bins),
                        channel_number=chan,
                    )
                )
                for chan in list(channels)
            }

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=0,
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=0,
                )
            )
            azimuth = fixed_azimuths.pop()
            elevation = fixed_elevations.pop()
            if any([fixed_azimuths, fixed_elevations]):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations

            for chan in channels:
                region_chan = regions[str(chan)]

                stim_spikes = np.array(
                    rec.stim_spiketrains(channel_number=chan, ignore_onset=0.050),
                    dtype="object",
                )

                psths = trial_spiketrains[chan]
                spiketrains_by_level = group_by_param(stim_spikes, trial_levels)
                psths_by_level = group_by_param(psths, trial_levels)
                for level, level_spiketrains in spiketrains_by_level.items():
                    level_phaselocking = get_phaselocking_am_stimuli(
                        level_spiketrains,
                        modulation_frequency=modulation_frequency,
                    )

                    mean_psth = np.mean(psths_by_level[level], axis=0)
                    time_firstspike, time_peak = get_latency(
                        mean_psth[latency_bins[:-1] >= delay],
                        latency_bins[latency_bins >= delay],
                    )
                    time_firstspike = time_firstspike - delay
                    time_peak = time_peak - delay

                    # if (
                    #     (time_firstspike < 0.0)
                    #     or (time_peak < 0.0)
                    #     or (time_firstspike > 0.040)
                    #     or (time_peak > 0.040)
                    # ):
                    #     continue

                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "azimuth": azimuth,
                        "elevation": elevation,
                        "intensity": level,
                        "modulation_frequency": modulation_frequency,
                        "singlestim_plv": level_phaselocking["vector_strength"],
                        "singlestim_plv_angle": level_phaselocking["mean_phase"],
                        "singlestim_plv_p": level_phaselocking["p"],
                        "psth": mean_psth,
                        "first_spike_latency": time_firstspike,
                        "max_peak_latency": time_peak,
                        "hemisphere": hemisphere,
                        "stimtype": "singlestim",
                        "region": region_chan,
                    }
                    singlestim_phaselocking.append(tmp)
    singlestim_phaselocking_df = pd.DataFrame(singlestim_phaselocking)
    return singlestim_phaselocking_df


def twostim_ccg(df, data_dir):
    twostim_ccg = []
    for session in iter_session_values(
        df, filename_filter=filter_relative_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            regions = rec.global_parameters()["regions"]

            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            fixed_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["fixed_index"]
                )
            )
            fixed_modulation_frequency = fixed_modulation_frequencies.pop()
            if fixed_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del fixed_modulation_frequencies

            # Excluded if owl struggled during RLF with driver f_AM:
            if (session["date"], session["owl"]) in struggled_during_am[
                fixed_modulation_frequency
            ]:
                continue

            varying_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["varying_index"]
                )
            )
            varying_modulation_frequency = varying_modulation_frequencies.pop()
            if varying_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del varying_modulation_frequencies

            psth_bins = rec.aggregate_stim_params(stim_psth_bins)
            base_bins = rec.aggregate_stim_params(base_psth_bins)

            fixed_level: float = fixed_varying["fixed_value"]  # type: ignore
            varying_levels: npt.NDArray[np.float64] = np.array(
                rec.aggregate_stim_params(
                    stim_level, stimulus_index=fixed_varying["varying_index"]
                )
            )
            unique_levels: list[float] = fixed_varying["varying_values"]  # type: ignore
            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            varying_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            varying_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            fixed_azimuth = fixed_azimuths.pop()
            fixed_elevation = fixed_elevations.pop()
            varying_azimuth = varying_azimuths.pop()
            varying_elevation = varying_elevations.pop()
            if any(
                [fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations]
            ):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations

            # Binary spiketrains during and before stimuli for all channels
            stim_spiketrains: dict[int, npt.NDArray[np.int_]] = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, psth_bins, channel_number=chan
                    )
                )
                for chan in channels
            }
            base_spiketrains = {
                chan: np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, base_bins, channel_number=chan
                    )
                )
                for chan in channels
            }

            if (hemisphere == "left" and varying_azimuth >= 0) or (
                hemisphere == "right" and varying_azimuth <= 0
            ):
                ipsi_contra = "ipsi"
            else:
                ipsi_contra = "contra"

            if ipsi_contra == "contra":
                # Don't save when competitor is on not-represented side
                continue

            for i, (chan1, chan2) in enumerate(combinations(channels, 2)):
                region_1 = regions[str(chan1)]
                region_2 = regions[str(chan2)]

                if region_1 == region_2:
                    if (region_1 == "OT") & (region_2 == "OT"):
                        corr_type = "within_OT"
                    elif (region_1 == "Forebrain") & (region_2 == "Forebrain"):
                        corr_type = "within_Forebrain"
                else:
                    corr_type = "cross_region"

                if chan1 < 17:
                    depth_order = [
                        14,
                        11,
                        3,
                        6,
                        2,
                        7,
                        1,
                        8,
                        4,
                        5,
                        16,
                        9,
                        15,
                        10,
                        13,
                        12,
                    ]
                elif chan1 >= 17:
                    depth_order = [
                        30,
                        27,
                        19,
                        22,
                        18,
                        23,
                        17,
                        24,
                        20,
                        21,
                        32,
                        25,
                        31,
                        26,
                        29,
                        28,
                    ]

                chan1_index = depth_order.index(chan1)

                if chan2 < 17:
                    depth_order = [
                        14,
                        11,
                        3,
                        6,
                        2,
                        7,
                        1,
                        8,
                        4,
                        5,
                        16,
                        9,
                        15,
                        10,
                        13,
                        12,
                    ]
                elif chan2 >= 17:
                    depth_order = [
                        30,
                        27,
                        19,
                        22,
                        18,
                        23,
                        17,
                        24,
                        20,
                        21,
                        32,
                        25,
                        31,
                        26,
                        29,
                        28,
                    ]

                chan2_index = depth_order.index(chan2)

                depth_distance = abs(chan1_index - chan2_index)

                for level in unique_levels:
                    # Boolean array to select trials of this level:
                    mask = varying_levels == level
                    # Data for this condition:
                    psth_u1 = stim_spiketrains[chan1][mask]
                    psth_u2 = stim_spiketrains[chan2][mask]
                    base_u1 = base_spiketrains[chan1][mask]
                    base_u2 = base_spiketrains[chan2][mask]

                    lags = (
                        scipy.signal.correlation_lags(
                            psth_u1.shape[1], psth_u2.shape[1]
                        )
                        * 0.001
                    )

                    # Mean firing rates:
                    resp_rate_u1: float = psth_u1.sum() / psth_u1.shape[0]
                    resp_rate_u2: float = psth_u2.sum() / psth_u2.shape[0]
                    base_rate_u1: float = base_u1.sum() / base_u1.shape[0]
                    base_rate_u2: float = base_u2.sum() / base_u2.shape[0]
                    # Geometric means:
                    gm_resp_rate = (resp_rate_u1 * resp_rate_u2) ** 0.5
                    gm_base_rate = (base_rate_u1 * base_rate_u2) ** 0.5
                    # Exclude of levels with no response
                    if gm_resp_rate <= gm_base_rate:
                        continue

                    ccg, shuff_ccg = cross_correlation(psth_u1, psth_u2, lags)

                    # Normalize by (geometric) mean response rate and psth length
                    norm_ccg = (
                        (ccg - np.mean(ccg)) / (gm_resp_rate) / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])
                    norm_shuff = (
                        (shuff_ccg - np.mean(shuff_ccg))
                        / (gm_resp_rate)
                        / psth_u1.shape[1]
                    )  # ??? len(psth_bins[0])

                    smscorrected = norm_ccg - norm_shuff

                    ccg_peak = get_peak(
                        smscorrected, lags, lag_window=0.015, baseline_window=0.05
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.05
                    )
                    ccg_peak_uncorrected = get_peak(
                        norm_ccg, lags, lag_window=0.015, baseline_window=0.05
                    )

                    relative_level = level - fixed_level

                    if ccg_peak["peak_corr"] > 5 * ccg_peak["baseline_std"]:
                        tmp = {
                            "date": session["date"],
                            "owl": session["owl"],
                            "channel1": chan1,
                            "channel2": chan2,
                            "fixedazi": fixed_azimuth,
                            "fixedele": fixed_elevation,
                            "fixedintensity": fixed_level,
                            "fixed_modulation_frequency": fixed_modulation_frequency,
                            "varyingazi": varying_azimuth,
                            "varyingele": varying_elevation,
                            "varyingintensity": level,
                            "varying_modulation_frequency": varying_modulation_frequency,
                            "relative_level": relative_level,
                            "xcorr_peak": ccg_peak["peak_corr"],
                            "peak_time": ccg_peak["peak_lag"],
                            "synchrony_val": ccg_peak["peak_area"],
                            "xcorr_width": ccg_peak["peak_width"],
                            "xcorr_peak_shuff": ccg_peak_shuff["peak_corr"],
                            "peak_time_shuff": ccg_peak_shuff["peak_lag"],
                            "synchrony_val_shuff": ccg_peak_shuff["peak_area"],
                            "xcorr_width_shuff": ccg_peak_shuff["peak_width"],
                            "stimlocked_peak": ccg_peak_uncorrected["peak_corr"],
                            "stim_locked_peak_time": ccg_peak_uncorrected["peak_lag"],
                            "stimlocked_synchrony_val": ccg_peak_uncorrected[
                                "peak_area"
                            ],
                            "hemisphere": hemisphere,
                            "ipsi_contra": ipsi_contra,
                            "stimtype": "twostim",
                            "ccg": smscorrected,
                            "ccg_shuff": norm_shuff,
                            "ccg_uncorrected": norm_ccg,
                            "corr_type": corr_type,
                            "depth_distance": depth_distance,
                        }
                        twostim_ccg.append(tmp)

    twostim_ccg_df = pd.DataFrame(twostim_ccg)
    return twostim_ccg_df


def twostim_rlf(df, data_dir):
    twostim_rlf = []
    for session in iter_session_values(
        df, filename_filter=filter_relative_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            fixed_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["fixed_index"]
                )
            )
            fixed_modulation_frequency = fixed_modulation_frequencies.pop()
            if fixed_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del fixed_modulation_frequencies

            # Excluded if owl struggled during RLF with driver f_AM:
            if (session["date"], session["owl"]) in struggled_during_am[
                fixed_modulation_frequency
            ]:
                continue

            varying_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["varying_index"]
                )
            )
            varying_modulation_frequency = varying_modulation_frequencies.pop()
            if varying_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del varying_modulation_frequencies

            fixed_level: float = fixed_varying["fixed_value"]  # type: ignore
            varying_levels = np.array(
                rec.aggregate_stim_params(
                    stim_level, stimulus_index=fixed_varying["varying_index"]
                )
            )

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            varying_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            varying_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            fixed_azimuth = fixed_azimuths.pop()
            fixed_elevation = fixed_elevations.pop()
            varying_azimuth = varying_azimuths.pop()
            varying_elevation = varying_elevations.pop()
            if any(
                [fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations]
            ):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations

            if (hemisphere == "left" and varying_azimuth >= 0) or (
                hemisphere == "right" and varying_azimuth <= 0
            ):
                ipsi_contra = "ipsi"
            else:
                ipsi_contra = "contra"

            if ipsi_contra == "contra":
                # Don't save when competitor is on not-represented side
                continue

            for chan in channels:
                region_chan = regions[str(chan)]
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                resp_by_level = group_by_param(resp, varying_levels)
                for level, level_resp in resp_by_level.items():
                    relative_level = level - fixed_level
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "fixed_modulation_frequency": fixed_modulation_frequency,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "varying_modulation_frequency": varying_modulation_frequency,
                        "relative_level": relative_level,
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "hemisphere": hemisphere,
                        "stimtype": "twostim",
                        "region": region_chan,
                    }
                    twostim_rlf.append(tmp)
    twostim_rlf_df = pd.DataFrame(twostim_rlf)
    return twostim_rlf_df


def twostim_gamma_power(df, data_dir):
    twostim_gamma_power = []
    for session in iter_session_values(
        df, filename_filter=filter_relative_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            fixed_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["fixed_index"]
                )
            )
            fixed_modulation_frequency = fixed_modulation_frequencies.pop()
            if fixed_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del fixed_modulation_frequencies

            # Excluded if owl struggled during RLF with driver f_AM:
            if (session["date"], session["owl"]) in struggled_during_am[
                fixed_modulation_frequency
            ]:
                continue

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]

            varying_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["varying_index"]
                )
            )
            varying_modulation_frequency = varying_modulation_frequencies.pop()
            if varying_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del varying_modulation_frequencies

            fixed_level: float = fixed_varying["fixed_value"]  # type: ignore
            varying_levels = np.array(
                rec.aggregate_stim_params(
                    stim_level, stimulus_index=fixed_varying["varying_index"]
                )
            )

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            varying_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            varying_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            fixed_azimuth = fixed_azimuths.pop()
            fixed_elevation = fixed_elevations.pop()
            varying_azimuth = varying_azimuths.pop()
            varying_elevation = varying_elevations.pop()
            if any(
                [fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations]
            ):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations

            if (hemisphere == "left" and varying_azimuth >= 0) or (
                hemisphere == "right" and varying_azimuth <= 0
            ):
                ipsi_contra = "ipsi"
            else:
                ipsi_contra = "contra"

            if ipsi_contra == "contra":
                # Don't save when competitor is on not-represented side
                continue

            for chan in channels:
                region_chan = regions[str(chan)]
                stim_spikes = np.array(
                    rec.stim_spiketrains(channel_number=chan, ignore_onset=0.050),
                    dtype="object",
                )
                spiketrains_by_level = group_by_param(stim_spikes, varying_levels)

                lfp_arr = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan)
                )
                lfp_arr = lfp_arr - np.mean(lfp_arr, axis=0)
                bandpass_lfp = bandpass_filter(
                    lfp_arr,
                    20,
                    50,
                    fs=lfp_samplingrate,
                    order=1,
                )

                trial_power = np.array(
                    [
                        get_power(
                            bandpass_lfp[trial_index],
                            baseline_start=trial_delays[trial_index]
                            - trial_durations[trial_index],
                            baseline_stop=trial_delays[trial_index] - 0.05,
                            stim_start=trial_delays[trial_index] + 0.05,
                            stim_stop=trial_delays[trial_index]
                            + trial_durations[trial_index],
                            samplingrate=lfp_samplingrate,
                        )
                        for trial_index in rec.powlfile.trial_indexes
                    ]
                )
                power_by_level = group_by_param(trial_power, varying_levels)
                lfp_by_level = group_by_param(bandpass_lfp, varying_levels)

                for level, level_power in power_by_level.items():
                    relative_level = level - fixed_level
                    level_phaselocking = get_phaselocking(
                        lfp_by_level[level], spiketrains_by_level[level]
                    )
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "fixed_modulation_frequency": fixed_modulation_frequency,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "varying_modulation_frequency": varying_modulation_frequency,
                        "relative_level": relative_level,
                        "gammapower": np.mean(level_power),
                        "gammapower_sem": scipy.stats.sem(level_power),
                        "gamma_plv": level_phaselocking["vector_strength"],
                        "gamma_plv_angle": level_phaselocking["mean_phase"],
                        "gamma_plv_p": level_phaselocking["p"],
                        "hemisphere": hemisphere,
                        "stimtype": "twostim",
                        "region": region_chan,
                    }

                    twostim_gamma_power.append(tmp)
    twostim_gamma_power_df = pd.DataFrame(twostim_gamma_power)
    return twostim_gamma_power_df


def twostim_phaselocking(df, data_dir):
    twostim_phaselocking = []

    for session in iter_session_values(
        df, filename_filter=filter_relative_level_am, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            # Exclude early recordings with ramp inside stimulus duration:
            stim_duration = set(rec.aggregate_stim_params(stim_len)).pop()
            if stim_duration < 1.010:
                continue

            fixed_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["fixed_index"]
                )
            )
            fixed_modulation_frequency = fixed_modulation_frequencies.pop()
            if fixed_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del fixed_modulation_frequencies

            # Excluded if owl struggled during RLF with driver f_AM:
            if (session["date"], session["owl"]) in struggled_during_am[
                fixed_modulation_frequency
            ]:
                continue

            varying_modulation_frequencies = set(
                rec.aggregate_stim_params(
                    stim_f_am, stimulus_index=fixed_varying["varying_index"]
                )
            )
            varying_modulation_frequency = varying_modulation_frequencies.pop()
            if varying_modulation_frequencies:
                raise ValueError("Modulation frequency varying unexpectedly")
            del varying_modulation_frequencies

            fixed_level: float = fixed_varying["fixed_value"]  # type: ignore
            varying_levels = np.array(
                rec.aggregate_stim_params(
                    stim_level, stimulus_index=fixed_varying["varying_index"]
                )
            )

            fixed_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            fixed_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["fixed_index"],
                )
            )
            varying_azimuths = set(
                rec.aggregate_stim_params(
                    lambda params: params["azi"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            varying_elevations = set(
                rec.aggregate_stim_params(
                    lambda params: params["ele"],
                    stimulus_index=fixed_varying["varying_index"],
                )
            )
            fixed_azimuth = fixed_azimuths.pop()
            fixed_elevation = fixed_elevations.pop()
            varying_azimuth = varying_azimuths.pop()
            varying_elevation = varying_elevations.pop()
            if any(
                [fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations]
            ):
                raise ValueError("Stimuli are not at the same positions for all trials")
            del fixed_azimuths, fixed_elevations, varying_azimuths, varying_elevations

            if (hemisphere == "left" and varying_azimuth >= 0) or (
                hemisphere == "right" and varying_azimuth <= 0
            ):
                ipsi_contra = "ipsi"
            else:
                ipsi_contra = "contra"

            if ipsi_contra == "contra":
                # Don't save when competitor is on not-represented side
                continue

            for chan in channels:
                region_chan = regions[str(chan)]
                stim_spikes = np.array(
                    rec.stim_spiketrains(channel_number=chan, ignore_onset=0.050),
                    dtype="object",
                )
                spiketrains_by_level = group_by_param(stim_spikes, varying_levels)

                for level, level_spiketrain in spiketrains_by_level.items():
                    relative_level = level - fixed_level
                    fixed_level_phaselocking = get_phaselocking_am_stimuli(
                        level_spiketrain,
                        modulation_frequency=fixed_modulation_frequency,
                    )
                    varying_level_phaselocking = get_phaselocking_am_stimuli(
                        level_spiketrain,
                        modulation_frequency=varying_modulation_frequency,
                    )
                    delta_angles = (
                        varying_level_phaselocking["spike_angles"]
                        - fixed_level_phaselocking["spike_angles"]
                    )
                    phase_differences = np.arctan2(
                        np.sin(delta_angles), np.cos(delta_angles)
                    )
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "fixed_modulation_frequency": fixed_modulation_frequency,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "varying_modulation_frequency": varying_modulation_frequency,
                        "relative_level": relative_level,
                        "fixedstim_plv": fixed_level_phaselocking["vector_strength"],
                        "fixedstim_plv_angle": fixed_level_phaselocking["mean_phase"],
                        "fixed_stim_angles": fixed_level_phaselocking["spike_angles"],
                        "fixedstim_plv_p": fixed_level_phaselocking["p"],
                        "varyingstim_plv": varying_level_phaselocking[
                            "vector_strength"
                        ],
                        "varyingstim_plv_angle": varying_level_phaselocking[
                            "mean_phase"
                        ],
                        "varying_stim_angles": varying_level_phaselocking[
                            "spike_angles"
                        ],
                        "varyingstim_plv_p": varying_level_phaselocking["p"],
                        "stim_phase_differences": phase_differences,
                        "hemisphere": hemisphere,
                        "stimtype": "twostim",
                        "region": region_chan,
                    }

                    twostim_phaselocking.append(tmp)

    twostim_phaselocking_df = pd.DataFrame(twostim_phaselocking)
    return twostim_phaselocking_df


if __name__ == "__main__":
    raise SystemExit(main())
