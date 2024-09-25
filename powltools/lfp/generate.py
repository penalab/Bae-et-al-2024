from functools import partial
import numpy as np
import numpy.typing as npt
from ..filters.helpers import chain_filters
from ..filters.offlinefilters import bandpass_filter, lowpass_filter, notch_filter


def filter_lfp(
    signal: npt.NDArray[np.float64],
    spikeindexes: npt.NDArray[np.int_],
    samplingrate: float,
) -> npt.NDArray[np.float64]:
    # Filter settings
    notch_filters = chain_filters(
        partial(notch_filter, fs=samplingrate, w0=60.0, Q=1.0),
        partial(notch_filter, fs=samplingrate, w0=180.0, Q=1.0),
        name="notch_filters",
    )
    lfp_bandpass_filter = partial(
        bandpass_filter, lowcut=300.0, highcut=4700.0, fs=samplingrate, order=2
    )
    lfp_lowpass_filter = partial(
        lowpass_filter, highcut=200.0, fs=samplingrate, order=2
    )
    # Remove 60 Hz and 180 Hz humming frequencies:
    signal = notch_filters(signal)
    # Split signal into `spikeband` and `nonspikeband`
    spikeband = lfp_bandpass_filter(signal)
    nonspikeband = signal - spikeband
    # uses spiketime indices to subtract and replace field potential with NaNs
    # exactly 2 msec before and after a spiketime
    win = int(samplingrate * 0.002)
    for spikeindex in spikeindexes:
        spikeband[spikeindex - win : spikeindex + win] = np.nan
    # linearly interpolates around each NaN in field potential signal
    nans = np.isnan(spikeband)
    # Replace nans in lfp by interpolated values:
    spikeband[nans] = np.interp(np.where(nans)[0], np.where(~nans)[0], spikeband[~nans])

    # Add rlf and spikeband back together and final lowpass filter:
    signal = lfp_lowpass_filter(nonspikeband + spikeband)

    return signal
