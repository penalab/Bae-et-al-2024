"""Some functions to make my life easier. Not actually part of data analysis.
"""

from textwrap import dedent
from typing import Optional, TypedDict, cast
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import BoxStyle
from matplotlib.transforms import ScaledTranslation, blended_transform_factory
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
from scikit_posthocs import posthoc_tukey

bc_colors = {100: "green", 40: "orange", 20: "red"}
einstein_blue_light = "#9acae8"
einstein_blue_dark = "#18397a"


def reload(module_or_function):
    """Reload a module or the module a function was defined in"""
    import importlib, sys
    from types import ModuleType, FunctionType

    # if it's a function or class, find the module it was defined in:
    if (
        hasattr(module_or_function, "__module__")
        and module_or_function.__module__ != "__main__"
    ):
        module_or_function = sys.modules[module_or_function.__module__]
    if isinstance(module_or_function, ModuleType):
        # We have a module now that we CAN reload:
        reloaded_module = importlib.reload(module_or_function)
        print(f"Reloaded module {reloaded_module.__name__}")
        return
    raise ValueError(
        f"Cannot reload whatever {module_or_function!r} is."
        " Must be a module or a function defined in one"
    )


def make_dimensions(
    n_rows=1,
    n_cols=1,
    individual_width_inch=3.0,
    wspace_inch=0.5,
    left_inch=2.0,
    right_inch=0.5,
    individual_height_inch=3.0,
    hspace_inch=0.5,
    bottom_inch=1.0,
    top_inch=0.5,
):
    """Make dimensions (figsize and gridspec_kw) for a subplotted figure from absolute values"""
    wspace = wspace_inch / individual_width_inch
    plot_area_width_inch = individual_width_inch * n_cols + (n_cols - 1) * wspace_inch
    total_width_inch = plot_area_width_inch + left_inch + right_inch
    left = left_inch / total_width_inch
    right = 1 - right_inch / total_width_inch

    hspace = hspace_inch / individual_height_inch
    plot_area_height_inch = individual_height_inch * n_rows + (n_rows - 1) * hspace_inch
    total_height_inch = plot_area_height_inch + bottom_inch + top_inch
    bottom = bottom_inch / total_height_inch
    top = 1 - top_inch / total_height_inch

    return (total_width_inch, total_height_inch), dict(
        left=left, right=right, wspace=wspace, bottom=bottom, top=top, hspace=hspace
    )


def make_figure(
    n_rows=1,
    n_cols=1,
    individual_width_inch=3.0,
    wspace_inch=0.5,
    left_inch=2.0,
    right_inch=0.5,
    individual_height_inch=3.0,
    hspace_inch=0.5,
    bottom_inch=1.0,
    top_inch=0.5,
    sharex=False,
    sharey=False,
):

    dimensions_dict = dict(
        n_rows=n_rows,
        n_cols=n_cols,
        individual_width_inch=individual_width_inch,
        wspace_inch=wspace_inch,
        left_inch=left_inch,
        right_inch=right_inch,
        individual_height_inch=individual_height_inch,
        hspace_inch=hspace_inch,
        bottom_inch=bottom_inch,
        top_inch=top_inch,
    )

    figsize, gridspec_kw = make_dimensions(**dimensions_dict)

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        gridspec_kw=gridspec_kw,
        sharex=sharex,
        sharey=sharey,
        # make sure, axs is ALWAYS a 2D-array:
        squeeze=False,
    )
    fig.dimensions_dict = dimensions_dict
    fig.dimensions_dict["w"] = figsize[0]
    fig.dimensions_dict["h"] = figsize[1]
    # Create Axes for global axis labels
    if axs.size > 1:
        axg = fig.add_subplot(axs[0, 0].get_gridspec()[:], zorder=-100)
        plt.setp(axg, frame_on=False, xticks=[], yticks=[])
    else:
        axg = axs[0, 0]
    axg.set_zorder(20)
    return fig, axs, axg


def unit_colors(s):
    colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
    return {ku: colors[kku % len(colors)] for kku, ku in enumerate(s.units.keys())}


def vmarker(ax, x, above_axis=True, **plot_kwargs):
    plot_kwargs = {
        **dict(
            markersize=10,
            marker="v",
            mfc="w",
            mec="k",
            zorder=10,
        ),
        **plot_kwargs,
    }
    trans = mpl.transforms.blended_transform_factory(
        ax.transData, ax.transAxes
    ) + mpl.transforms.ScaledTranslation(
        0,
        (1 if above_axis else -1) * plot_kwargs["markersize"] / 2 / 72.0,
        ax.figure.dpi_scale_trans,
    )
    return ax.plot(x, 0, transform=trans, clip_on=False, **plot_kwargs)


mpl.axes.Axes.vmarker = vmarker


def vmarker_left(ax, x, above_axis=True, **plot_kwargs):
    plot_kwargs = {
        **dict(
            markersize=10,
            marker="v",
            mfc="w",
            mec="k",
            zorder=10,
        ),
        **plot_kwargs,
    }
    trans = mpl.transforms.blended_transform_factory(
        ax.transAxes, ax.transData
    ) + mpl.transforms.ScaledTranslation(
        (1 if above_axis else -1) * plot_kwargs["markersize"] / 2 / 72.0,
        0,
        ax.figure.dpi_scale_trans,
    )
    return ax.plot(0, x, transform=trans, clip_on=False, **plot_kwargs)


mpl.axes.Axes.vmarker_left = vmarker_left


def make_cont_x(*values, before=50, after=50, n=100):
    left = np.min([np.min(v) for v in values]) - before
    right = np.max([np.max(v) for v in values]) + after
    return np.linspace(left, right, n)


class AnovaTukeyResults(TypedDict):
    fstat: float
    anova_p: float
    tukey: pd.DataFrame


def anova_tukey(
    df: pd.DataFrame, val_col: str, group_col: str, title: Optional[str] = None
) -> AnovaTukeyResults:
    ## ANOVA one-way
    groupdata = [
        cast(pd.Series, df.loc[groups, val_col]).values for groups in df.index.unique()
    ]

    fstat, pval = scipy.stats.f_oneway(*groupdata, axis=0)
    posthoc = posthoc_tukey(df.reset_index(), val_col=val_col, group_col=group_col)

    print(
        dedent(
            f"""
            ANOVA TUKEY STATS {title or ''}
            {fstat = :.2f}, {pval = :.3g}
            {posthoc}
            """
        )
    )

    return {"fstat": fstat, "anova_p": pval, "tukey": posthoc}


def figure_add_axes_inch(
    fig: Figure,
    left=None,
    width=None,
    right=None,
    bottom=None,
    height=None,
    top=None,
    label=None,
    **kwargs,
) -> Axes:
    """Add Axes to a figure with inch coordinates."""
    # Check number of arguments:
    n_horz_args = sum([left is not None, width is not None, right is not None])
    if not n_horz_args == 2:
        raise ValueError(
            f"Need exactly 2 horizontal arguments, but {n_horz_args} given."
        )
    n_vert_args = sum([bottom is not None, height is not None, top is not None])
    if not n_vert_args == 2:
        raise ValueError(
            f"Need exactly 2 horizontal arguments, but {n_vert_args} given."
        )
    # Unique label:
    if label is None:
        label = f"ax{len(fig.get_axes()):02}"

    # Figure dimensions:
    fig_w, fig_h = fig.get_size_inches()

    # Horizontal:
    if right is None:
        l = left / fig_w
        w = width / fig_w
    elif width is None:
        l = left / fig_w
        w = (fig_w - left - right) / fig_w
    else:  # left is None
        w = width / fig_w
        l = (fig_w - right - width) / fig_w

    # Vertical:
    if top is None:
        b = bottom / fig_h
        h = height / fig_h
    elif height is None:
        b = bottom / fig_h
        h = (fig_h - bottom - top) / fig_h
    else:  # bottom is None
        h = height / fig_h
        b = (fig_h - top - height) / fig_h

    return fig.add_axes((l, b, w, h), label=label, **kwargs)


def figure_add_axes_group_inch(
    fig: Figure,
    nrows=1,
    ncols=1,
    group_top=0.2,
    group_left=0.8,
    individual_width=1.2,
    individual_height=0.8,
    wspace=0.1,
    hspace=0.1,
):
    axs = []
    for kr in range(nrows):
        axs.append([])
        for kc in range(ncols):
            ax = figure_add_axes_inch(
                fig,
                top=group_top + kr * (individual_height + hspace),
                height=individual_height,
                left=group_left + kc * (individual_width + wspace),
                width=individual_width,
            )
            axs[-1].append(ax)
    axs = np.asarray(axs)
    axg = figure_add_axes_inch(
        fig,
        top=group_top,
        height=nrows * individual_height + (nrows - 1) * hspace,
        left=group_left,
        width=ncols * individual_width + (ncols - 1) * wspace,
    )
    plt.setp(axg, frame_on=False, xticks=[], yticks=[], zorder=20)
    return axs, axg


def subplot_indicator(ax: Axes, label=None, fontsize=16, pad_inch=None, **kwargs):
    shift_left = (fontsize / 2 / 72.0) if pad_inch is None else pad_inch
    trans = ax.transAxes + ScaledTranslation(
        -shift_left, +0 / 72.0, ax.figure.dpi_scale_trans  # type: ignore
    )
    if label is None:
        label = ax.get_label()
    if "ha" in kwargs:
        kwargs["horizontalalignment"] = kwargs.pop("ha")
    if "va" in kwargs:
        kwargs["verticalalignment"] = kwargs.pop("va")
    textkwargs = dict(
        horizontalalignment="right",
        verticalalignment="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )
    textkwargs.update(kwargs)
    ax.text(0.0, 1.0, label, transform=trans, **textkwargs)  # type: ignore


def plot_bracket(
    ax: Axes,
    left: float,
    right: float,
    text: str = "",
    y: float = 0.8,
    shrink: float = 0.8,
):
    m = (left + right) / 2
    d = right - left
    left = m - (d / 2) * shrink
    right = m + (d / 2) * shrink
    ax.plot(
        [left, left, right, right],
        [y - 0.05, y, y, y - 0.05],
        ls="-",
        lw=1.0,
        color="k",
        transform=(
            blended_transform_factory(ax.transData, ax.transAxes)
            + ScaledTranslation(0, 0, ax.figure.dpi_scale_trans)  # type: ignore
        ),
    )
    if text:
        ax.text(
            m,
            y,
            text,
            ha="center",
            va="bottom",
            fontsize=8,
            transform=(
                blended_transform_factory(ax.transData, ax.transAxes)
                + ScaledTranslation(0, 2 / 72, ax.figure.dpi_scale_trans)  # type: ignore
            ),
        )


def condition_batch(
    fig: Figure,
    left: float = 0.8,
    top: float = 0.3,
    text: str = "label",
    *,
    fontsize: float | int = 8,
    color="black",
    ha="right",
    va="bottom",
    x_pt=0.0,
    y_pt=0.0,
    pad_pt=4.0,
):
    x_pt = x_pt + pad_pt
    y_pt = y_pt + pad_pt
    fig.text(
        0,
        1,
        text,
        color="white",
        fontsize=fontsize,
        fontweight="bold",
        ha=ha,
        va=va,
        bbox={
            "facecolor": color,
            "alpha": 1,
            "linewidth": 0,
            "boxstyle": BoxStyle(
                "Round",
                pad=pad_pt / fontsize,
                rounding_size=2 * pad_pt / fontsize,
            ),
        },
        transform=fig.transFigure  # type: ignore
        + ScaledTranslation(
            +left + (-1 if ha == "right" else +1) * x_pt / 72,
            -top + (-1 if va == "top" else +1) * y_pt / 72,
            fig.dpi_scale_trans,  # type: ignore
        ),
    )
