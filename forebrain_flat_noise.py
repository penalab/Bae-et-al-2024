from itertools import combinations, repeat
import pathlib
import numpy as np
import numpy.typing as npt
import scipy.stats
import scipy.signal
import pandas as pd
import random
from powltools.io.file import POwlFile
from powltools.analysis.recording import Recording
from powltools.analysis.recording import stim_position
from powltools.analysis.recording import stim_level
from powltools.analysis.recording import stim_delay
from powltools.analysis.recording import stim_len
from powltools.analysis.recording import group_by_param
from powltools.analysis.recording import group_by_multiparam
from powltools.filters.offlinefilters import bandpass_filter

from xcorr_tools import binary_spiketrain
from xcorr_tools import make_psth_bins
from xcorr_tools import stim_psth_bins
from xcorr_tools import base_psth_bins
from xcorr_tools import cross_correlation
from xcorr_tools import get_peak
from xcorr_tools import fixed_var_stimuli
from xcorr_tools import iter_session_values
from xcorr_tools import filter_rate_level_flat
from xcorr_tools import filter_rate_level_flat_out
from xcorr_tools import filter_relative_level_flat
from xcorr_tools import filter_rate_level_out_flat
from xcorr_tools import filter_relative_level_switch
from xcorr_tools import filter_srf
from xcorr_tools import get_power
from xcorr_tools import get_phaselocking
from xcorr_tools import get_phaselocking_binned
from xcorr_tools import get_phaselocking_old


def main():
    # Single stimulus data, flat noise, curated/clean
    # Contains one line per [date, owl, channel, level]
    OUTDIR = pathlib.Path("./forebrain_intermediate_results").absolute()
    OUTDIR.mkdir(exist_ok=True)

    data_dir = r"E:\Andrea-Freefield"
    df = pd.read_csv(r"./pooled_data_excel/auditory_units_combined.csv")
    df.set_index(["date", "owl"], inplace=True)
    df.sort_index(inplace=True)
    region = "dualregion"

    # Single stim:
    print("srf".upper())
    single_srf_df = srf(df, data_dir)
    single_srf_df.to_feather(OUTDIR / f"single_srf_{region}.feather")

    print("singlestim_ccg".upper())
    single_ccg_df = singlestim_ccg(df, data_dir)
    single_ccg_df.to_feather(OUTDIR / f"single_ccg_{region}.feather")

    print("singlestim_ccg_out".upper())
    single_ccg_df_out = singlestim_ccg_out(df, data_dir)
    single_ccg_df_out.to_feather(OUTDIR / f"single_ccg_{region}_out.feather")

    print("singlestim_rlf".upper())
    singlestim_rlf_df = singlestim_rlf(df, data_dir)
    singlestim_rlf_df.to_feather(OUTDIR / f"single_rlf_{region}.feather")

    print("singlestim_rlf_out".upper())
    singlestim_rlf_df_out = singlestim_rlf_out(df, data_dir)
    singlestim_rlf_df_out.to_feather(OUTDIR / f"single_rlf_out_{region}.feather")

    print("singlestim_gamma_power".upper())
    singlestim_gamma_power_df = singlestim_gamma_power(df, data_dir)
    singlestim_gamma_power_df.to_feather(
        OUTDIR / f"single_gamma_power_{region}.feather"
    )

    # Two Stim:
    print("twostim_ccg".upper())
    twostim_ccg_df = twostim_ccg(df, data_dir)
    twostim_ccg_df.to_feather(OUTDIR / f"twostim_ccg_{region}.feather")

    print("twostim_ccg_switch".upper())
    twostim_ccg_switch_df = twostim_ccg_switch(df, data_dir)
    twostim_ccg_switch_df.to_feather(OUTDIR / f"twostim_ccg_switch_{region}.feather")

    print("twostim_rlf".upper())
    twostim_rlf_df = twostim_rlf(df, data_dir)
    twostim_rlf_df.to_feather(OUTDIR / f"twostim_rlf_{region}.feather")

    print("twostim_gamma_power".upper())
    twostim_gamma_power_df = twostim_gamma_power(df, data_dir)
    twostim_gamma_power_df.to_feather(OUTDIR / f"twostim_gamma_power_{region}.feather")

    print("cross_region_gamma_phase".upper())
    cross_region_phase = cross_region_gamma_phase(df, data_dir)
    cross_region_phase.to_feather(OUTDIR / f"cross_region_gamma_{region}.feather")

    print("twostim_sfc".upper())
    sfc_within_df = within_area_sfc_competition(df, data_dir)
    sfc_within_df.to_feather(OUTDIR / f"twostim_sfc_within_area_{region}.feather")

    print("twostim_rlf_switch".upper())
    twostim_rlf_df_switch = twostim_rlf_switch(df, data_dir)
    twostim_rlf_df_switch.to_feather(OUTDIR / f"twostim_rlf_switch_{region}.feather")

    print("cross_region_sfc".upper)
    sfc_cross_region_df = cross_region_sfc(df, data_dir)
    sfc_cross_region_df.to_feather(OUTDIR / f"sfc_cross_region_df_{region}.feather")

    return 0


def get_latency(trace, time_bins):
    max_ind = np.argmax(trace)
    half_ind = np.argmax(trace >= trace[max_ind] / 2)
    latency = time_bins[half_ind]
    peak_time = time_bins[max_ind]
    return latency, peak_time


def srf(df, data_dir):
    srf_df = []
    for session in iter_session_values(
        df, filename_filter=filter_srf, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            trial_delays = set(rec.aggregate_stim_params(stim_delay, stimulus_index=0))
            trial_durations = set(rec.aggregate_stim_params(stim_len, stimulus_index=0))
            delay = trial_delays.pop()
            duration = trial_durations.pop()
            if any([trial_delays, trial_durations]):
                raise ValueError(
                    "Stimulus delay or duration not the same for all trials"
                )
            del trial_delays, trial_durations
            trial_levels = np.array(
                rec.aggregate_stim_params(stim_level, stimulus_index=0)
            )

            for chan in channels:
                region_chan = regions[str(chan)]
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                max_norm_resp = resp / np.max(resp)

                trial_positions = np.array(
                    rec.aggregate_stim_params(stim_position, stimulus_index=0)
                )

                resp_by_position = group_by_multiparam(resp, trial_positions)
                norm_resp_by_position = group_by_multiparam(
                    max_norm_resp, trial_positions
                )

                for (azi, ele), rep_resp in resp_by_position.items():
                    mean_resp = np.mean(rep_resp)
                    norm_mean = np.mean(norm_resp_by_position[(azi, ele)])
                    sigma_squared = np.std(rep_resp) ** 2
                    if norm_mean == 0:
                        fano_factor = 1
                    else:
                        fano_factor = sigma_squared / mean_resp

                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "intensity": trial_levels[0],
                        "azimuth": azi,
                        "elevation": ele,
                        "position": tuple([azi, ele]),
                        "resp": mean_resp,
                        "norm_resp": norm_mean,
                        "fano_factor": fano_factor,
                        "region": region_chan,
                        "hemisphere": hemisphere,
                    }
                    srf_df.append(tmp)
    srf_df = pd.DataFrame(srf_df)
    return srf_df


def singlestim_ccg(df, data_dir):
    singlestim_ccg = []

    for session in iter_session_values(
        df, filename_filter=filter_rate_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

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
                        smscorrected, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_uncorrected = get_peak(
                        norm_ccg, lags, lag_window=0.015, baseline_window=0.050
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


def singlestim_ccg_out(df, data_dir):
    singlestim_ccg = []

    for session in iter_session_values(
        df, filename_filter=filter_rate_level_flat_out, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

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
                    if (region_1 == "OT") and (region_2 == "OT"):
                        corr_type = "within_OT"
                    elif (region_1 == "Forebrain") and (region_2 == "Forebrain"):
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
                        smscorrected, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_uncorrected = get_peak(
                        norm_ccg, lags, lag_window=0.015, baseline_window=0.050
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
        df, filename_filter=filter_rate_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

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
                max_norm_resp = resp / np.max(resp)
                trial_levels = np.array(
                    rec.aggregate_stim_params(stim_level, stimulus_index=0)
                )
                resp_by_level = group_by_param(resp, trial_levels)
                norm_resp_by_level = group_by_param(max_norm_resp, trial_levels)

                psth_by_level = group_by_param(trial_spiketrains[chan], trial_levels)
                # print(psth)
                # print(psth_by_level)
                for level, level_resp in resp_by_level.items():
                    mean_psth = np.mean(psth_by_level[level], axis=0)
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
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "norm_resp": np.mean(norm_resp_by_level[level]),
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
        df, filename_filter=filter_rate_level_out_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

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
                max_norm_resp = resp / np.max(resp)

                resp_by_level = group_by_param(resp, trial_levels)
                norm_resp_by_level = group_by_param(max_norm_resp, trial_levels)

                psth_by_level = group_by_param(trial_spiketrains[chan], trial_levels)
                # print(psth)
                # print(psth_by_level)
                for level, level_resp in resp_by_level.items():
                    mean_psth = np.mean(psth_by_level[level], axis=0)
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
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "norm_resp": np.mean(norm_resp_by_level[level]),
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
        df, filename_filter=filter_rate_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

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

                lfp_arr = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan)
                )
                lfp_arr = lfp_arr - np.mean(lfp_arr, axis=0)

                bands = {"lowgamma": [20, 50], "highgamma": [50, 75]}
                for key, vals in bands.items():
                    bandpass_lfp = bandpass_filter(
                        lfp_arr,
                        vals[0],
                        vals[1],
                        fs=lfp_samplingrate,
                        order=2,
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
                    spiketrains_by_level = group_by_param(stim_spikes, trial_levels)
                    for level, level_power in power_by_level.items():

                        level_phaselocking = get_phaselocking_old(
                            lfp_by_level[level], spiketrains_by_level[level]
                        )

                        level_phaselocking_shuffled = get_phaselocking_old(
                            lfp_by_level[level],
                            spiketrains_by_level[level][
                                random.sample(list(np.arange(0, 20)), 20)
                            ],
                        )

                        tmp = {
                            "date": session["date"],
                            "owl": session["owl"],
                            "channel": chan,
                            "azimuth": azimuth,
                            "elevation": elevation,
                            "intensity": level,
                            "gammapower": np.mean(level_power),
                            "gammapower_sem": scipy.stats.sem(level_power),
                            "gamma_plv": level_phaselocking["vector_strength"],
                            "gamma_plv_angle": level_phaselocking["mean_phase"],
                            "gamma_spike_angles": level_phaselocking["spike_angles"],
                            "gamma_plv_p": level_phaselocking["p"],
                            "gamma_plv_shuffled": level_phaselocking_shuffled[
                                "vector_strength"
                            ],
                            "gamma_plv_angle_shuffled": level_phaselocking_shuffled[
                                "mean_phase"
                            ],
                            "gamma_spike_angles_shuffled": level_phaselocking_shuffled[
                                "spike_angles"
                            ],
                            "gamma_plv_p_shuffled": level_phaselocking_shuffled["p"],
                            "hemisphere": hemisphere,
                            "stimtype": "singlestim",
                            "region": region_chan,
                            "gamma_band": key,
                        }
                        singlestim_gamma_power.append(tmp)
    singlestim_gamma_power_df = pd.DataFrame(singlestim_gamma_power)
    return singlestim_gamma_power_df


def twostim_ccg(df, data_dir):
    twostim_ccg = []

    for session in iter_session_values(
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
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
                    if (region_1 == "OT") and (region_2 == "OT"):
                        corr_type = "within_OT"
                    elif (region_1 == "Forebrain") and (region_2 == "Forebrain"):
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
                        smscorrected, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_uncorrected = get_peak(
                        norm_ccg, lags, lag_window=0.015, baseline_window=0.050
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
                            "varyingazi": varying_azimuth,
                            "varyingele": varying_elevation,
                            "varyingintensity": level,
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


def twostim_ccg_switch(df, data_dir):
    twostim_ccg = []

    for session in iter_session_values(
        df, filename_filter=filter_relative_level_switch, data_dir=data_dir
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
                    if (region_1 == "OT") and (region_2 == "OT"):
                        corr_type = "within_OT"
                    elif (region_1 == "Forebrain") and (region_2 == "Forebrain"):
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
                        smscorrected, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_shuff = get_peak(
                        norm_shuff, lags, lag_window=0.015, baseline_window=0.050
                    )
                    ccg_peak_uncorrected = get_peak(
                        norm_ccg, lags, lag_window=0.015, baseline_window=0.050
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
                            "varyingazi": varying_azimuth,
                            "varyingele": varying_elevation,
                            "varyingintensity": level,
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
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            delay = trial_delays.pop()
            duration = trial_durations.pop()
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

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

            for chan in channels:
                psth_by_level = group_by_param(trial_spiketrains[chan], varying_levels)
                region_chan = regions[str(chan)]
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                resp_by_level = group_by_param(resp, varying_levels)
                for level, level_resp in resp_by_level.items():
                    mean_psth = np.mean(psth_by_level[level], axis=0)
                    relative_level = level - fixed_level
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "relative_level": relative_level,
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "psth": mean_psth,
                        "hemisphere": hemisphere,
                        "stimtype": "twostim",
                        "region": region_chan,
                    }
                    twostim_rlf.append(tmp)

    twostim_rlf_df = pd.DataFrame(twostim_rlf)
    return twostim_rlf_df


def twostim_rlf_switch(df, data_dir):
    twostim_rlf = []

    for session in iter_session_values(
        df, filename_filter=filter_relative_level_switch, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            delay = trial_delays.pop()
            duration = trial_durations.pop()
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

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

            for chan in channels:
                psth_by_level = group_by_param(trial_spiketrains[chan], varying_levels)
                region_chan = regions[str(chan)]
                resp = rec.response_rates(channel_number=chan, stimulus_index=0)
                resp_by_level = group_by_param(resp, varying_levels)
                for level, level_resp in resp_by_level.items():
                    mean_psth = np.mean(psth_by_level[level], axis=0)
                    relative_level = level - fixed_level
                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "relative_level": relative_level,
                        "resp": np.mean(level_resp),
                        "sem": scipy.stats.sem(level_resp),
                        "psth": mean_psth,
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
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
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

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]

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

                lfp_arr = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan)
                )
                lfp_arr = lfp_arr - np.mean(lfp_arr, axis=0)

                bands = {"lowgamma": [20, 50], "highgamma": [50, 75]}

                for key, vals in bands.items():
                    bandpass_lfp = bandpass_filter(
                        lfp_arr,
                        vals[0],
                        vals[1],
                        fs=lfp_samplingrate,
                        order=2,
                    )

                    #                 bandpass_lfp = bandpass_filter(
                    #                     lfp_arr,
                    #                     20,
                    #                     50,
                    #                     fs=lfp_samplingrate,
                    #                     order=2,
                    #                 )

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
                    spiketrains_by_level = group_by_param(stim_spikes, varying_levels)

                    for level, level_power in power_by_level.items():
                        relative_level = level - fixed_level
                        level_phaselocking = get_phaselocking_old(
                            lfp_by_level[level], spiketrains_by_level[level]
                        )
                        level_phaselocking_shuffled = get_phaselocking_old(
                            lfp_by_level[level],
                            spiketrains_by_level[level][
                                random.sample(list(np.arange(0, 20)), 20)
                            ],
                        )

                        tmp = {
                            "date": session["date"],
                            "owl": session["owl"],
                            "channel": chan,
                            "fixedazi": fixed_azimuth,
                            "fixedele": fixed_elevation,
                            "fixedintensity": fixed_level,
                            "varyingazi": varying_azimuth,
                            "varyingele": varying_elevation,
                            "varyingintensity": level,
                            "relative_level": relative_level,
                            "gammapower": np.mean(level_power),
                            "gammapower_sem": scipy.stats.sem(level_power),
                            "gamma_plv": level_phaselocking["vector_strength"],
                            "gamma_plv_angle": level_phaselocking["mean_phase"],
                            "gamma_spike_angles": level_phaselocking["spike_angles"],
                            "gamma_plv_p": level_phaselocking["p"],
                            "gamma_plv_shuffled": level_phaselocking_shuffled[
                                "vector_strength"
                            ],
                            "gamma_plv_angle_shuffled": level_phaselocking_shuffled[
                                "mean_phase"
                            ],
                            "gamma_spike_angles_shuffled": level_phaselocking_shuffled[
                                "spike_angles"
                            ],
                            "gamma_plv_p_shuffled": level_phaselocking_shuffled["p"],
                            "hemisphere": hemisphere,
                            "stimtype": "twostim",
                            "region": region_chan,
                            "gamma_band": key,
                        }

                        twostim_gamma_power.append(tmp)

    twostim_gamma_power_df = pd.DataFrame(twostim_gamma_power)
    return twostim_gamma_power_df


def within_area_sfc_competition(df, data_dir):
    lfp_data = []

    for session in iter_session_values(
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
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

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]
            psth_bins = rec.aggregate_stim_params(stim_psth_bins)

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
                region_1 = regions[str(chan)]
                lfp_arr_chan = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan)
                )
                induced_lfp_chan = lfp_arr_chan[:, 1050:2000] - np.mean(
                    lfp_arr_chan[:, 1050:2000], axis=0
                )
                stim_spikes_chan1 = np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, psth_bins, channel_number=chan
                    )
                )

                lfp_by_level = group_by_param(induced_lfp_chan, varying_levels)
                spikes_by_level = group_by_param(stim_spikes_chan1, varying_levels)

                for level, chan1_lfp in lfp_by_level.items():
                    relative_level = level - fixed_level

                    f, Cxy = scipy.signal.coherence(
                        chan1_lfp,
                        spikes_by_level[level],
                        lfp_samplingrate,
                        nperseg=190,
                        detrend="constant",
                    )
                    mean_coherence = np.nanmean(Cxy, axis=0)

                    f, Cxy = scipy.signal.coherence(
                        chan1_lfp,
                        stim_spikes_chan1[
                            random.sample(
                                list(np.arange(len(spikes_by_level[level]))), 20
                            )
                        ],
                        lfp_samplingrate,
                        nperseg=190,
                        noverlap=140,
                        detrend="constant",
                    )
                    mean_coherence_rd = np.nanmean(Cxy, axis=0)

                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel": chan,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "relative_level": relative_level,
                        "frequency": f,
                        "spike_field_coherence": mean_coherence,
                        "trial_shuffled_sfc": mean_coherence_rd,
                        "stimtype": "twostim",
                        "region": region_1,
                    }
                    lfp_data.append(tmp)
    lfp_data = pd.DataFrame(lfp_data)
    return lfp_data


def cross_region_sfc(df, data_dir):
    # def twostim_gamma_power(df, data_dir):
    coherence_df = []

    for session in iter_session_values(
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        #     if session['date'] != '2024-04-20':
        #         continue

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]
            psth_bins = rec.aggregate_stim_params(stim_psth_bins)

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

            for i, (chan1, chan2) in enumerate(combinations(channels, 2)):
                region_1 = regions[str(chan1)]
                region_2 = regions[str(chan2)]

                if region_1 == region_2:
                    if (region_1 == "OT") and (region_2 == "OT"):
                        corr_type = "within_OT"
                    elif (region_1 == "Forebrain") and (region_2 == "Forebrain"):
                        corr_type = "within_Forebrain"
                else:
                    corr_type = "cross_region"

                if corr_type != "cross_region":
                    continue

                stim_spikes_chan1 = np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, psth_bins, channel_number=chan1
                    )
                )
                stim_spikes_chan2 = np.vstack(
                    rec.aggregrate_spikes(
                        binary_spiketrain, psth_bins, channel_number=chan2
                    )
                )

                lfp_arr_chan1 = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan1)
                )
                induced_lfp_chan1 = lfp_arr_chan1[:, 1050:2000] - np.mean(
                    lfp_arr_chan1[:, 1050:2000], axis=0
                )

                lfp_arr_chan2 = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan2)
                )
                induced_lfp_chan2 = lfp_arr_chan2[:, 1050:2000] - np.mean(
                    lfp_arr_chan2[:, 1050:2000], axis=0
                )

                lfp_by_level_chan1 = group_by_param(induced_lfp_chan1, varying_levels)
                lfp_by_level_chan2 = group_by_param(induced_lfp_chan2, varying_levels)

                chan1_spiketrains_by_level = group_by_param(
                    stim_spikes_chan1, varying_levels
                )
                chan2_spiketrains_by_level = group_by_param(
                    stim_spikes_chan2, varying_levels
                )

                for level, chan1_lfp in lfp_by_level_chan1.items():
                    relative_level = level - fixed_level

                    f, Cxy = scipy.signal.coherence(
                        chan1_lfp,
                        lfp_by_level_chan2[level],
                        lfp_samplingrate,
                        nperseg=190,
                        noverlap=140,
                        detrend="constant",
                    )
                    mean_coherence = np.nanmean(Cxy, axis=0)

                    f, Cxy_rd = scipy.signal.coherence(
                        chan1_lfp,
                        lfp_by_level_chan2[level][
                            random.sample(list(np.arange(0, 20)), 20)
                        ],
                        lfp_samplingrate,
                        nperseg=190,
                        noverlap=140,
                        detrend="constant",
                    )
                    mean_coherence_rd = np.nanmean(Cxy_rd, axis=0)

                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel1": chan1,
                        "channel2": chan2,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "relative_level": relative_level,
                        "frequency": f,
                        "lfp_coherence": mean_coherence,
                        "lfp_coherence_random": mean_coherence_rd,
                        "stimtype": "twostim",
                        "corr_type": corr_type,
                    }
                    coherence_df.append(tmp)
    coherence_df = pd.DataFrame(coherence_df)
    return coherence_df


def cross_region_gamma_phase(df, data_dir):
    twostim_gamma_phase = []

    for session in iter_session_values(
        df, filename_filter=filter_relative_level_flat, data_dir=data_dir
    ):
        print(
            f"{session['date']} {session['owl']} ({len(session['filenames'])} files, {len(session['channels'])} channels)"
        )

        #     if session['date'] != '2024-04-20':
        #         continue

        channels = session["channels"]

        for filename in session["filenames"]:
            rec = Recording(POwlFile(filename))
            fixed_varying = fixed_var_stimuli(rec, stim_level)
            hemisphere: str = rec.global_parameters()["session"]["hemisphere"]  # type: ignore
            regions = rec.global_parameters()["regions"]

            trial_delays = rec.aggregate_stim_params(stim_delay, stimulus_index=0)
            trial_durations = rec.aggregate_stim_params(stim_len, stimulus_index=0)
            lfp_samplingrate = rec.global_parameters()["signals"]["lfp"]["samplingrate"]

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

            for i, (chan1, chan2) in enumerate(combinations(channels, 2)):
                region_1 = regions[str(chan1)]
                region_2 = regions[str(chan2)]

                if region_1 == region_2:
                    if (region_1 == "OT") and (region_2 == "OT"):
                        corr_type = "within_OT"
                    elif (region_1 == "Forebrain") and (region_2 == "Forebrain"):
                        corr_type = "within_Forebrain"
                else:
                    corr_type = "cross_region"

                if corr_type != "cross_region":
                    continue

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

                stim_spikes_chan1 = np.array(
                    rec.stim_spiketrains(channel_number=chan1, ignore_onset=0.050),
                    dtype="object",
                )
                stim_spikes_chan2 = np.array(
                    rec.stim_spiketrains(channel_number=chan2, ignore_onset=0.050),
                    dtype="object",
                )

                lfp_arr_chan1 = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan1)
                )
                induced_lfp_chan1 = lfp_arr_chan1 - np.mean(lfp_arr_chan1, axis=0)

                lfp_arr_chan2 = np.vstack(
                    rec.aggregrate_lfps(lambda lfp: lfp, channel_number=chan2)
                )
                induced_lfp_chan2 = lfp_arr_chan2 - np.mean(lfp_arr_chan2, axis=0)

                # gammaband = {'lowgamma': [20,50], 'highgamma':[50,75]}

                # for key, vals in gammaband.items():
                bandpass_lfp_chan1 = bandpass_filter(
                    induced_lfp_chan1,
                    20,
                    50,
                    fs=lfp_samplingrate,
                    order=2,
                )

                bandpass_lfp_chan2 = bandpass_filter(
                    induced_lfp_chan2,
                    20,
                    50,
                    fs=lfp_samplingrate,
                    order=2,
                )
                lfp_by_level_chan1 = group_by_param(bandpass_lfp_chan1, varying_levels)
                lfp_by_level_chan2 = group_by_param(bandpass_lfp_chan2, varying_levels)

                chan1_spiketrains_by_level = group_by_param(
                    stim_spikes_chan1, varying_levels
                )
                chan2_spiketrains_by_level = group_by_param(
                    stim_spikes_chan2, varying_levels
                )

                for level, chan1_spikes in chan1_spiketrains_by_level.items():
                    relative_level = level - fixed_level

                    chan1_lfp_chan1spikes = get_phaselocking_old(
                        lfp_by_level_chan1[level], chan1_spiketrains_by_level[level]
                    )

                    chan2_lfp_chan2spikes = get_phaselocking_old(
                        lfp_by_level_chan2[level], chan2_spiketrains_by_level[level]
                    )

                    cross_chan1lfp_chan2spikes = get_phaselocking_old(
                        lfp_by_level_chan1[level], chan2_spiketrains_by_level[level]
                    )
                    cross_chan2lfp_chan1spikes = get_phaselocking_old(
                        lfp_by_level_chan2[level], chan1_spiketrains_by_level[level]
                    )

                    chan1lfp_chan2spikes_shuff = get_phaselocking_old(
                        lfp_by_level_chan1[level],
                        chan2_spiketrains_by_level[level][
                            random.sample(list(np.arange(0, 20)), 20)
                        ],
                    )

                    chan2lfp_chan1spikes_shuff = get_phaselocking_old(
                        lfp_by_level_chan2[level],
                        chan1_spiketrains_by_level[level][
                            random.sample(list(np.arange(0, 20)), 20)
                        ],
                    )

                    tmp = {
                        "date": session["date"],
                        "owl": session["owl"],
                        "channel1": chan1,
                        "channel2": chan2,
                        "fixedazi": fixed_azimuth,
                        "fixedele": fixed_elevation,
                        "fixedintensity": fixed_level,
                        "varyingazi": varying_azimuth,
                        "varyingele": varying_elevation,
                        "varyingintensity": level,
                        "relative_level": relative_level,
                        "nrt_meanangle": chan2_lfp_chan2spikes["mean_phase"],
                        "nrt_vector_strength": chan2_lfp_chan2spikes["vector_strength"],
                        "ot_meanangle": chan1_lfp_chan1spikes["mean_phase"],
                        "ot_vector_strength": chan1_lfp_chan1spikes["vector_strength"],
                        "otlfp_nrtspikes_vs": cross_chan1lfp_chan2spikes[
                            "vector_strength"
                        ],
                        "otlfp_nrtspikes_meanangle": cross_chan1lfp_chan2spikes[
                            "mean_phase"
                        ],
                        "otlfp_nrtspikes_angle": cross_chan1lfp_chan2spikes[
                            "spike_angles"
                        ],
                        "otlfp_nrtspikes_pvalue": cross_chan1lfp_chan2spikes["p"],
                        "otlfp_nrtspikes_vs_shuffle": chan1lfp_chan2spikes_shuff[
                            "vector_strength"
                        ],
                        "otlfp_nrtspikes_meanangle_shuffle": chan1lfp_chan2spikes_shuff[
                            "mean_phase"
                        ],
                        "otlfp_nrtspikes_angle_shuffle": chan1lfp_chan2spikes_shuff[
                            "spike_angles"
                        ],
                        "otlfp_nrtspikes_pvalue_shuffle": chan1lfp_chan2spikes_shuff[
                            "p"
                        ],
                        "nrtlfp_otspikes_vs": cross_chan2lfp_chan1spikes[
                            "vector_strength"
                        ],
                        "nrtlfp_otspikes_meanangle": cross_chan2lfp_chan1spikes[
                            "mean_phase"
                        ],
                        "nrtlfp_otspikes_angle": cross_chan2lfp_chan1spikes[
                            "spike_angles"
                        ],
                        "nrtlfp_otspikes_pvalue": cross_chan2lfp_chan1spikes["p"],
                        "nrtlfp_otspikes_vs_shuffle": chan2lfp_chan1spikes_shuff[
                            "vector_strength"
                        ],
                        "nrtlfp_otspikes_meanangle_shuffle": chan2lfp_chan1spikes_shuff[
                            "mean_phase"
                        ],
                        "nrtlfp_otspikes_angle_shuffle": chan2lfp_chan1spikes_shuff[
                            "spike_angles"
                        ],
                        "nrtlfp_otspikes_pvalue_shuffle": chan2lfp_chan1spikes_shuff[
                            "p"
                        ],
                        "hemisphere": hemisphere,
                        "stimtype": "twostim",
                        "corr_type": corr_type,
                    }
                    twostim_gamma_phase.append(tmp)
    twostim_gamma_phase = pd.DataFrame(twostim_gamma_phase)
    return twostim_gamma_phase


if __name__ == "__main__":
    raise SystemExit(main())
