from __future__ import annotations

from typing import Callable, Optional
import numpy as np
import numpy.typing as npt

SignalType = npt.NDArray[np.float64]
FilterType = Callable[[SignalType], SignalType]


def chain_filters(
    *filters: tuple[FilterType, ...],
    name: Optional[str] = None,
) -> FilterType:
    def filter_chain(signal: SignalType) -> SignalType:
        for filter in filters:
            signal = filter(signal)
        return signal

    if name is not None:
        filter_chain.__name__ = name
        filter_chain.__qualname__ = name
        filter_chain.__module__ = None

    return filter_chain
