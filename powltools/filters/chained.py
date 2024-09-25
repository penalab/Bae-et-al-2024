from __future__ import annotations

from functools import partial, lru_cache
from .helpers import chain_filters, FilterType, SignalType
from .offlinefilters import notch_filter, highpass_filter


@lru_cache(maxsize=4)
def get_spikecontinuous_filter(
    fs: float,
    *,
    notch_frequencies: list[float] = [60.0, 180.0],
    highpass_cutoff: float = 300.0,
) -> FilterType:
    return chain_filters(
        *(partial(notch_filter, fs=fs, w0=f, Q=1.0) for f in notch_frequencies),
        partial(highpass_filter, lowcut=highpass_cutoff, fs=fs, order=2),
        name="spikecontinuous_filter",
    )
