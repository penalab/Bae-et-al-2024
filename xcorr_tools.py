from __future__ import annotations as _annotations

import os
import glob
from functools import cache
import re
from typing import Callable, Iterable, Iterator, TypedDict
import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.signal
from powltools.analysis.recording import (
    Recording,
    SpikeTimesType,
    StimParamFunc,
    StimulusParams,
    stim_delay,
    stim_len,
    stim_level,
)


@cache
def make_psth_bins(
    stimdelay: float,
    stimduration: float,
    binsize: float = 0.001,
    offset: float = 0.050,
) -> npt.NDArray[np.float64]:
    return np.arange(stimdelay + offset, stimdelay + stimduration + binsize, binsize)


@cache
def make_psth_bins_base(
    stimdelay: float,
    stimduration: float,
    binsize: float = 0.001,
    offset: float = 0.050,
) -> npt.NDArray[np.float64]:
    return np.arange(stimdelay - stimduration, stimdelay - offset + binsize, binsize)


def stim_psth_bins(stim_params: StimulusParams) -> npt.NDArray[np.float64]:
    return make_psth_bins(
        stim_delay(stim_params), stim_len(stim_params), binsize=0.001, offset=0.050
    )


def base_psth_bins(stim_params: StimulusParams) -> npt.NDArray[np.float64]:
    return make_psth_bins_base(
        stim_delay(stim_params), stim_len(stim_params), binsize=0.001, offset=0.050
    )


def trial_psth_bins(stim_params: StimulusParams) -> npt.NDArray[np.float64]:
    return make_psth_bins(
        0, stim_len(stim_params) + stim_delay(stim_params), binsize=0.001, offset=0.00
    )


def binary_spiketrain(
    spikes: SpikeTimesType, bins: npt.NDArray[np.float64]
) -> npt.NDArray[np.int_]:
    return np.histogram(spikes, bins)[0].astype(bool).astype(int)


### From Andrea


def smooth(signal: npt.NDArray[np.float64], window_size: int = 5):
    """Smooth a signal with a moving rectangular window.

    Implements equivalent functionality MATLAB's smooth(a, window_size, 'moving')

    On both ends equivalently, average with a symmetric window as large as possible:

    out[0] = a[0]
    out[1] = sum(a[0:3]) / 3
    out[2] = sum(a[0:5]) / 5
    ...

    Parameters
    ----------
    a: NDArray
        1-D array containing the data to be smoothed
    window_size : int
        Smoothing window size needs, which must be odd number,
        as in the original MATLAB implementation

    Returns
    -------
    NDArray
        The smoothed signal

    Notes
    -----
    Implementation adapted from: https://stackoverflow.com/a/40443565
    """
    s = np.convolve(signal, np.ones(window_size, dtype="float"), "valid") / window_size
    r = np.arange(1, window_size - 1, 2)
    start = np.cumsum(signal[: window_size - 1])[::2] / r
    stop = (np.cumsum(signal[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((start, s, stop))


def cross_correlation(
    cond_psth: Iterable[npt.NDArray[np.float64 | np.int_]],
    cond_psth2: Iterable[npt.NDArray[np.float64 | np.int_]],
    lags,
):
    xcorr = np.asarray(
        [
            (
                scipy.signal.correlate(psth_ot, psth_fix)
                if np.any(psth_ot)
                else np.zeros(lags.size)
            )
            for psth_ot, psth_fix in zip(cond_psth, cond_psth2)
        ]
    )

    zip(cond_psth, cond_psth2[1:] + [cond_psth2[0]])

    shuffled_ind = np.arange(1, len(cond_psth) - 1)
    xcorr_shuff = np.asarray(
        [
            (
                scipy.signal.correlate(psth_ot, psth_fix)
                if np.any(psth_ot)
                else np.zeros(lags.size)
            )
            for psth_ot, psth_fix in zip(cond_psth[shuffled_ind], cond_psth2[:-1])
        ]
    )

    mean_xcorr = smooth(np.nanmean(xcorr, axis=0), 9)
    mean_shuff = smooth(np.nanmean(xcorr_shuff, axis=0), 9)

    return mean_xcorr, mean_shuff


def get_peak(
    ccg: npt.NDArray[np.float64],
    lags: npt.NDArray[np.float64],
    lag_window: float = 0.015,
    baseline_window: float = 0.050,
) -> CCGpeakData:
    """Get the relevant peak values of a cross-correlogram (CCG)

    Search for the peak value is restricted to lags within [-lag_window, +lag_window], inclusive.

    Parameters
    ----------
    ccg : array
        The cross-correlogram, like returned from scipy.signal.correlate
    lags : array
        The corresponding lags, like returned from scipy.signal.correlation_lags
    lags_window : float
        Defines the range of lags (around 0) where the peak is allowed, same unit as lags
    baseline_window : float
        Defines the range of lags (from both end) to retrieve the CCG baseline

    Returns
    -------
    dict
        peak_corr
            The peak correlation value, from which the baseline mean was already subtracted
        peak_lag
            The lag at which the peak was found
        baseline_mean
            Mean of the baseline
        baseline_std
            Standard deviation of the baseline
        peak_width
            Half-height width of the
        peak_area
            Area under peak, within half-height width
    """
    window_start = np.searchsorted(lags, -lag_window)
    window_stop = np.searchsorted(lags, +lag_window) + 1
    # Number of elements for baseline on each end:
    baseline_n = int(np.unique(np.round(baseline_window / np.diff(lags))))
    baseline = np.concatenate((ccg[:baseline_n], ccg[-baseline_n:]))
    peak_ind = window_start + np.nanargmax(ccg[window_start:window_stop])
    baseline_mean: float = np.mean(baseline)

    # Half width and area under the curve
    halfmax = (ccg[peak_ind] + baseline_mean) / 2
    left_ind = peak_ind - np.argmax(ccg[peak_ind::-1] <= halfmax)
    right_ind = peak_ind + np.argmax(ccg[peak_ind::+1] <= halfmax)
    area_under_peak = np.trapz(ccg[left_ind:right_ind], x=lags[left_ind:right_ind])
    peak_width = lags[right_ind] - lags[left_ind]

    return {
        "peak_corr": ccg[peak_ind] - baseline_mean,
        "peak_lag": lags[peak_ind],
        "baseline_mean": baseline_mean,
        "baseline_std": np.std(baseline),
        "peak_area": area_under_peak,
        "peak_width": peak_width,
    }


def get_power(
    trial_lfp,
    baseline_start,
    baseline_stop,
    stim_start,
    stim_stop,
    samplingrate: float = 1000.0,
):
    s_hilb = np.abs(scipy.signal.hilbert(trial_lfp))
    hilb_power = s_hilb**2

    lfp_times = np.arange(trial_lfp.size) / samplingrate

    baseline_slice = slice(
        np.searchsorted(lfp_times, baseline_start),
        np.searchsorted(lfp_times, baseline_stop),
    )
    stim_slice = slice(
        np.searchsorted(lfp_times, stim_start), np.searchsorted(lfp_times, stim_stop)
    )
    baseline_pow = hilb_power[baseline_slice]
    stim_pow = hilb_power[stim_slice]

    baseline_rms = np.sqrt(np.sum(baseline_pow**2) / baseline_pow.size)
    stim_rms = np.sqrt(np.sum(stim_pow**2) / stim_pow.size)

    rel_db = 20 * np.log(stim_rms / baseline_rms)
    return rel_db


def wrap_to_pi(angle):
    """
    Wrap the angle to the range [-pi, pi].

    Parameters:
    angle (float or array-like): The angle(s) to be wrapped.

    Returns:
    wrapped_angle (float or ndarray): The wrapped angle(s).
    """
    wrapped_angle = (angle + np.pi) % (2 * np.pi) - np.pi
    return wrapped_angle


def get_phaselocking_old(filt_lfp, binarized_spikes):

    continuous_time = np.arange(0, filt_lfp.shape[1]) * 0.001
    continuous_time = np.asarray([round(val, 3) for val in continuous_time])

    s_hilb = scipy.signal.hilbert(filt_lfp)
    inst_phases = np.angle(s_hilb)

    spike_phase = []
    for ktrial, trial_spikes in enumerate(binarized_spikes):
        trial_phases = np.asarray(inst_phases[ktrial])
        for spike in trial_spikes:
            # spike_I = np.searchsorted(continuous_time, spike, side = 'right')
            spike_I = np.where(continuous_time <= spike)[0][-1]
            center = trial_phases[spike_I]
            spike_phase.append(center)
    spike_phase = np.asarray(spike_phase)

    # phase_locking_value= np.abs(np.sum(np.exp(np.dot(1j, spike_phase)))/len(spike_phase)) #fancy math

    # pi_bin = np.arange(-np.pi, np.pi + np.pi / 6, np.pi / 6)
    pi_bin = np.linspace(-np.pi, np.pi, 9)

    sfc = np.histogram(spike_phase, pi_bin)
    bins = sfc[1]

    ##calculate mean angle of frequency grouped data
    y_angle = sum(sfc[0] * np.sin(sfc[1][:-1])) / sum(sfc[0])
    x_angle = sum(sfc[0] * np.cos(sfc[1][:-1])) / sum(sfc[0])
    # r_val = np.sqrt((y_angle**2) + (x_angle**2))
    mean_ang = np.arctan2((y_angle), (x_angle))

    r_val = np.sqrt((y_angle**2) + (x_angle**2))
    d = np.diff(bins)[0]
    corr_fac = d / 2 / np.sin(d / 2)
    r_val = corr_fac * r_val

    ## Code for rayleigh significance
    total_n = np.sum(sfc[0])

    # compute Rayleigh's R (equ. 27.1)
    rayleigh_r = total_n * r_val

    # compute Rayleigh's z (equ. 27.2)
    z = rayleigh_r**2 / total_n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(
        np.sqrt(1 + 4 * total_n + 4 * (total_n**2 - rayleigh_r**2)) - (1 + 2 * total_n)
    )
    return {
        "spike_angles": spike_phase,
        "vector_strength": r_val,
        "mean_phase": mean_ang,
        "p": pval,
    }


def get_phaselocking(
    filt_lfp: Iterable[npt.NDArray[np.float64]],
    spiketrains: Iterable[SpikeTimesType],
    lfp_samplingrate: float = 1000.0,
) -> PhaseLockingData:
    filt_lfp = np.vstack(filt_lfp)
    lfp_times = np.arange(filt_lfp.shape[1]) / lfp_samplingrate

    s_hilb = scipy.signal.hilbert(filt_lfp)
    s_angles = np.angle(s_hilb)

    spike_angles = np.concatenate(
        [
            np.interp(trial_spiketimes, lfp_times, trial_phases)
            for trial_spiketimes, trial_phases in zip(spiketrains, s_angles)
        ]
    )

    spike_phases = np.exp(1j * np.asarray(spike_angles))
    z_mean = np.mean(spike_phases)
    mean_angle = np.angle(z_mean)
    vector_strength = np.abs(z_mean)

    #     spike_phases = np.cos(spike_angles) + np.sin(spike_angles) * 1j

    #     mean_vector = np.mean(spike_phases)
    #     mean_angle = np.angle(mean_vector)
    #     vector_strength = np.abs(mean_vector)

    ## Code for rayleigh significance
    total_n = spike_phases.size
    # compute Rayleigh's R (equ. 27.1)
    rayleigh_r = total_n * vector_strength
    # compute p value using approxation in Zar, p. 617
    pval = np.exp(
        np.sqrt(1 + 4 * total_n + 4 * (total_n**2 - rayleigh_r**2)) - (1 + 2 * total_n)
    )
    return {
        "spike_angles": spike_angles,
        "vector_strength": vector_strength,
        "mean_phase": mean_angle,
        "p": pval,
    }


def get_phaselocking_binned(
    filt_lfp: Iterable[npt.NDArray[np.float64]],
    spiketrains: Iterable[SpikeTimesType],
    lfp_samplingrate: float = 1000.0,
) -> PhaseLockingData:
    filt_lfp = np.vstack(filt_lfp)
    lfp_times = np.arange(filt_lfp.shape[1]) / lfp_samplingrate

    s_hilb = scipy.signal.hilbert(filt_lfp)
    s_angles = np.angle(s_hilb)

    spike_angles = np.concatenate(
        [
            np.interp(trial_spiketimes, lfp_times, trial_phases)
            for trial_spiketimes, trial_phases in zip(spiketrains, s_angles)
        ]
    )

    pi_bin = np.linspace(-np.pi, np.pi, 9)
    sfc = np.histogram(spike_angles, pi_bin)
    bins = sfc[1]

    ##calculate mean angle of frequency grouped data
    y_angle = sum(sfc[0] * np.sin(sfc[1][:-1])) / sum(sfc[0])
    x_angle = sum(sfc[0] * np.cos(sfc[1][:-1])) / sum(sfc[0])

    mean_ang = np.arctan2((y_angle), (x_angle))

    r_val = np.sqrt((y_angle**2) + (x_angle**2))
    d = np.diff(bins)[0]
    corr_fac = d / 2 / np.sin(d / 2)
    r_val = corr_fac * r_val

    ## Code for rayleigh significance
    total_n = np.sum(sfc[0])

    # compute Rayleigh's R (equ. 27.1)
    rayleigh_r = total_n * r_val

    # compute Rayleigh's z (equ. 27.2)
    z = rayleigh_r**2 / total_n

    # compute p value using approxation in Zar, p. 617
    pval = np.exp(
        np.sqrt(1 + 4 * total_n + 4 * (total_n**2 - rayleigh_r**2)) - (1 + 2 * total_n)
    )
    return {
        "spike_angles": spike_angles,
        "vector_strength": r_val,
        "mean_phase": mean_ang,
        "p": pval,
    }


def get_phaselocking_am_stimuli(
    spiketrains: Iterable[SpikeTimesType],
    modulation_frequency: float,
) -> PhaseLockingData:
    period = 1 / modulation_frequency
    spike_angles = np.concatenate(
        [
            (spiketrain % period) * modulation_frequency * 2 * np.pi
            for spiketrain in spiketrains
        ]
    )
    # Move values in between -pi to +pi
    spike_angles = spike_angles - np.pi

    spike_phases = np.cos(spike_angles) + np.sin(spike_angles) * 1j

    mean_vector = np.mean(spike_phases)
    mean_angle = np.angle(mean_vector)
    vector_strength = np.abs(mean_vector)

    ## Code for rayleigh significance
    total_n = spike_phases.size
    # compute Rayleigh's R (equ. 27.1)
    rayleigh_r = total_n * vector_strength
    # compute p value using approxation in Zar, p. 617
    pval = np.exp(
        np.sqrt(1 + 4 * total_n + 4 * (total_n**2 - rayleigh_r**2)) - (1 + 2 * total_n)
    )
    return {
        "spike_angles": spike_angles,
        "vector_strength": vector_strength,
        "mean_phase": mean_angle,
        "p": pval,
    }


def synchrony_ccg(ccg, lags):
    ind_left = np.where(lags[0] >= -0.010)[0][0]
    peak_ind = np.argmax(ccg[(lags[0] >= -0.010) & (lags[0] <= 0.010)])

    peak_ind = ind_left + peak_ind
    peakval = ccg[peak_ind]
    halfmax = peakval / 2
    left_border = np.where(ccg[:peak_ind] <= halfmax)[0][-1]
    right_border = np.where(ccg[peak_ind:] <= halfmax)[0][0]
    auc = np.trapz(ccg[left_border : right_border + peak_ind], dx=np.diff(lags[0])[0])
    width = lags[0][right_border + peak_ind] - lags[0][left_border]
    return auc, width


def filter_rate_level_flat(filename: str) -> bool:
    return (
        "rate_level" in filename
        and "amplitude" not in filename
        and "out" not in filename
        and "contra" not in filename
        and not "hi" in filename
        and not "lo" in filename
    )


def filter_rate_level_flat_out(filename: str) -> bool:
    return (
        "rate_level" in filename
        and "amplitude" not in filename
        and "out" in filename
        and "contra" not in filename
        and not "hi" in filename
        and not "lo" in filename
    )


def filter_rate_level_out_flat(filename: str) -> bool:
    return (
        "rate_level" in filename
        and "amplitude" not in filename
        and "out" in filename
        and not "hi" in filename
        and not "lo" in filename
    )


def filter_rate_level_contra_flat(filename: str) -> bool:
    return (
        "rate_level" in filename
        and "amplitude" not in filename
        and "contra" in filename
        and not "hi" in filename
        and not "lo" in filename
    )


def filter_relative_level_flat(filename: str) -> bool:
    filename = filename.lower()
    return (
        "relative_intensity" in filename
        and "contra" not in filename
        and "amplitude" not in filename
        and not "hi" in filename
        and not "lo" in filename
        and not "switch" in filename
    )


def filter_relative_level_switch(filename: str) -> bool:
    filename = filename.lower()
    return (
        "relative_intensity" in filename
        and "contra" not in filename
        and "amplitude" not in filename
        and not "hi" in filename
        and not "lo" in filename
        and "switch" in filename
    )


def filter_relative_level_contra_flat(filename: str) -> bool:
    filename = filename.lower()
    return (
        "relative_intensity" in filename
        and "contra" in filename
        and "amplitude" not in filename
        and not "hi" in filename
        and not "lo" in filename
    )


def filter_rate_level_am(filename: str) -> bool:
    return (
        "rate_level" in filename
        and "am" in filename
        and not "out" in filename
        and not "contra" in filename
    )


def filter_rate_level_out_am(filename: str) -> bool:
    return "rate_level" in filename and "am" in filename and "out" in filename


def filter_rate_level_contra_am(filename: str) -> bool:
    return "rate_level" in filename and "am" in filename and "contra" in filename


def filter_relative_level_am(filename: str) -> bool:
    return (
        "relative_intensity" in filename
        and "am" in filename
        and not "contra" in filename
    )


def filter_relative_level_contra_am(filename: str) -> bool:
    return (
        "relative_intensity" in filename and "am" in filename and "contra" in filename
    )


def filter_srf(filename: str) -> bool:
    return "spatial_receptive_field" in filename


# def filter_relative_level_am(filename: str) -> bool:
#    filename = filename.lower()
#    return bool(re.search("amplitude_?modulation_[12]", filename))


def iter_session_values(
    df: pd.DataFrame,
    filename_filter: Callable[[str], bool],
    data_dir: str,
) -> Iterator[SessionValues]:
    for index in sorted(set(df.index)):
        # = ('2023-05-15', 33)
        index_date, index_owl = index
        index_dirname = f"{index_date.replace('-', '')}_{index_owl}_awake"
        channels: list[int] = sorted(set(df.loc[index, "channel"]))  # type: ignore
        filenames = [
            filename
            for filename in glob.glob(os.path.join(data_dir, index_dirname, "*.h5"))
            if filename_filter(filename)
        ]
        yield {
            "date": index_date,
            "owl": index_owl,
            "dirname": index_dirname,
            "channels": channels,
            "filenames": filenames,
        }


def fixed_var_stimuli(
    rec: Recording, func: StimParamFunc = stim_level
) -> FixedVarStimuli:
    with rec.powlfile as pf:
        stimulus_indexes = {int(k) for k in pf["trials/0/stimuli"].keys()}
        if not len(stimulus_indexes) == 2:
            raise ValueError(
                "Only one stimulus found, expected two."
                if len(stimulus_indexes) == 1
                else "More than two stimuli found, expected two."
            )
        unique_values = {
            stimulus_index: set(
                rec.aggregate_stim_params(func, stimulus_index=stimulus_index)
            )
            for stimulus_index in stimulus_indexes
        }
    # Find the first stimulus_index with a single unique value
    fixed_index = next(
        stimulus_index
        for stimulus_index, uvalues in unique_values.items()
        if len(uvalues) == 1
    )
    varying_index = (stimulus_indexes - {fixed_index}).pop()
    return {
        "fixed_index": fixed_index,
        "fixed_value": unique_values[fixed_index].pop(),
        "varying_index": varying_index,
        "varying_values": sorted(unique_values[varying_index]),
    }


def spikes_to_phase(spiketrain, modulation_frequency):
    period = 1 / modulation_frequency
    phase = ((spiketrain % period) / period) * (2 * np.pi)
    phase = phase - np.pi
    return phase


## For type annotations:


class CCGpeakData(TypedDict):
    peak_corr: float
    peak_lag: float
    baseline_mean: float
    baseline_std: float
    peak_area: float
    peak_width: float


class SessionValues(TypedDict):
    date: str
    owl: int
    dirname: str
    channels: list[int]
    filenames: list[str]


class FixedVarStimuli(TypedDict):
    fixed_index: int
    fixed_value: str | int | float
    varying_index: int
    varying_values: list[str | int | float]


class PhaseLockingData(TypedDict):
    vector_strength: float
    mean_phase: float
    p: float
