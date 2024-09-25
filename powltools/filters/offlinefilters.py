"""Simple filters to be used in offline analysis

All filters use scipy.signal.filtfilt and, thus, avoid delaying the signal.
"""

import numpy as np
import scipy.signal
import numpy.typing as npt


def notch_filter(
    data: npt.NDArray,
    fs: float,
    w0: float = 60.0,
    Q: float = 30.0,
) -> npt.NDArray:
    """Second-order IIR notch filter

    Filters forward-backward with sosfiltfilt, which removes phase delays.
    This also means that the filter order will be twice that given here.

    Uses second-order sections (output="sos" and sosfiltfilt) following
    advice in the SciPy documentation to avoid numerical problems.

    Parameters
    ----------
    data : array_like
        The signal to be filtered
    fs : float
        Sampling frequency of `data`
    w0 : float
        Frequency to remove from a signal
    order : int
        Butterworth filter order. Note that resulting order will double.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `data`

    See Also
    --------
    scipy.signal.iirnotch : For filter description
    scipy.signal.filtfilt : For forward-backward filter application
    """
    # Generate filter parameters:
    d, e = scipy.signal.iirnotch(w0, Q, fs)
    # apply filter:
    y = scipy.signal.filtfilt(d, e, data)
    return y


def bandpass_filter(
    data: npt.NDArray[np.float64],
    lowcut: float,
    highcut: float,
    fs: float,
    order: int,
) -> npt.NDArray[np.float64]:
    """Butterworth bandpass filter

    Filters forward-backward with sosfiltfilt, which removes phase delays.
    This also means that the filter order will be twice that given here.

    Uses second-order sections (output="sos" and sosfiltfilt) following
    advice in the SciPy documentation to avoid numerical problems.

    Parameters
    ----------
    data : array_like
        The signal to be filtered
    lowcut : float
        Lower cutoff frequency
    highcut : float
        Higher cutoff frequency
    fs : float
        Sampling frequency of `data`
    order : int
        Butterworth filter order. Note that resulting order will double.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `data`

    See Also
    --------
    scipy.signal.butter : For filter description
    scipy.signal.sosfiltfilt : For forward-backward filter application
    """
    # Generate filter parameters:
    sos = scipy.signal.butter(
        order,
        [lowcut, highcut],
        btype="band",
        output="sos",
        fs=fs,
    )
    # apply filter:
    y = scipy.signal.sosfiltfilt(sos, data)
    return y


def highpass_filter(
    data: npt.NDArray[np.float64],
    lowcut: float,
    fs: float,
    order: int = 2,
) -> npt.NDArray[np.float64]:
    """Butterworth highpass filter

    Filters forward-backward with sosfiltfilt, which removes phase delays.
    This also means that the filter order will be twice that given here.

    Uses second-order sections (output="sos" and sosfiltfilt) following
    advice in the SciPy documentation to avoid numerical problems.

    Parameters
    ----------
    data : array_like
        The signal to be filtered
    lowcut : float
        Cutoff frequency
    fs : float
        Sampling frequency of `data`
    order : int
        Butterworth filter order. Note that resulting order will double.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `data`

    See Also
    --------
    scipy.signal.butter : For filter description
    scipy.signal.sosfiltfilt : For forward-backward filter application
    """
    # Generate filter parameters:
    sos = scipy.signal.butter(
        order,
        [lowcut],
        btype="highpass",
        output="sos",
        fs=fs,
    )
    # apply filter:
    y = scipy.signal.sosfiltfilt(sos, data)
    return y


def lowpass_filter(
    data: npt.NDArray[np.float64],
    highcut: float,
    fs: float,
    order: int = 2,
) -> npt.NDArray[np.float64]:
    """Butterworth lowpass filter

    Filters forward-backward with sosfiltfilt, which removes phase delays.
    This also means that the filter order will be twice that given here.

    Uses second-order sections (output="sos" and sosfiltfilt) following
    advice in the SciPy documentation to avoid numerical problems.

    Parameters
    ----------
    data : array_like
        The signal to be filtered
    highcut : float
        Cutoff frequency
    fs : float
        Sampling frequency of `data`
    order : int
        Butterworth filter order. Note that resulting order will double.

    Returns
    -------
    y : ndarray
        The filtered output with the same shape as `data`

    See Also
    --------
    scipy.signal.butter : For filter description
    scipy.signal.sosfiltfilt : For forward-backward filter application
    """
    # Generate filter parameters:
    sos = scipy.signal.butter(
        order,
        [highcut],
        btype="lowpass",
        output="sos",
        fs=fs,
    )
    # apply filter:
    y = scipy.signal.sosfiltfilt(sos, data)
    return y
