import pathlib
from typing import cast
import numpy as np
import pandas as pd


from powltools.io.file import POwlFile
from powltools.analysis.recording import (
    Recording,
    group_by_multiparam,
    stim_level,
    stim_position,
)
from xcorr_tools import fixed_var_stimuli


def main():
    OUTDIR = pathlib.Path("./forebrain_intermediate_results").absolute()
    OUTDIR.mkdir(exist_ok=True)
    data_dir = pathlib.Path(r"E:\Andrea-Freefield")
    # SRF data (Figure 1 A):
    df = srf_plot_data(
        filename=data_dir / "20240315_256_awake/01_spatial_receptive_field.h5",
        channel_number=30,
    )
    df.to_feather(OUTDIR / "example_srf_20240315_256.feather")

    # Coincident Plot Data
    # Figure 2 Forebrain (Flat)
    df = conincident_plot_data(
        filename=data_dir / "20240315_256_awake/03_relative_intensity.h5",
        channel_number1=30,
        channel_number2=28,
    )
    df.to_feather(OUTDIR / "example_spiketrains_forebrain_flat_20240315_256.feather")
    df = conincident_plot_data(
        filename=data_dir / "20240315_256_awake/03_relative_intensity.h5",
        channel_number1=30,
        channel_number2=19,
    )
    df.to_feather(
        OUTDIR / "example_spiketrains_forebrain_flat_different_20240315_256.feather"
    )
    # Figure 3C OT (Driver 55 Hz, Competitor 75 Hz)
    df = conincident_plot_data(
        filename=data_dir / "20240330_40_awake/03_relative_intensity_am_in55.h5",
        channel_number1=9,
        channel_number2=13,
    )
    df.to_feather(OUTDIR / "example_spiketrains_forebrain_driver55_20240330_40.feather")
    # Figure 3D OT (Driver 75 Hz, Competitor 55 Hz)
    df = conincident_plot_data(
        filename=data_dir / "20240330_40_awake/05_relative_intensity_am_in75.h5",
        channel_number1=9,
        channel_number2=13,
    )
    df.to_feather(OUTDIR / "example_spiketrains_forebrain_driver75_20240330_40.feather")

    ## for srf correlation example plots
    df = srf_plot_data(
        filename=data_dir / "20240420_54_awake/01_spatial_receptive_field.h5",
        channel_number=5,
    )
    df.to_feather(OUTDIR / "example_srf_lowcorr_OT_20240420_54.feather")

    df = srf_plot_data(
        filename=data_dir / "20240420_54_awake/01_spatial_receptive_field.h5",
        channel_number=17,
    )
    df.to_feather(OUTDIR / "example_srf_lowcorr_Forebrain_20240420_54.feather")

    df = srf_plot_data(
        filename=data_dir / "20240423_54_awake/01_spatial_receptive_field.h5",
        channel_number=5,
    )
    df.to_feather(OUTDIR / "example_srf_midlowcorr_OT_20240423_54.feather")

    df = srf_plot_data(
        filename=data_dir / "20240423_54_awake/01_spatial_receptive_field.h5",
        channel_number=19,
    )
    df.to_feather(OUTDIR / "example_srf_midlowcorr_Forebrain_20240423_54.feather")

    df = srf_plot_data(
        filename=data_dir / "20240420_54_awake/01_spatial_receptive_field.h5",
        channel_number=10,
    )
    df.to_feather(OUTDIR / "example_srf_midhighcorr_OT_20240420_54.feather")

    df = srf_plot_data(
        filename=data_dir / "20240420_54_awake/01_spatial_receptive_field.h5",
        channel_number=17,
    )
    df.to_feather(OUTDIR / "example_srf_midhighcorr_Forebrain_20240420_54.feather")

    df = srf_plot_data(
        filename=data_dir / "20240420_54_awake/01_spatial_receptive_field.h5",
        channel_number=15,
    )
    df.to_feather(OUTDIR / "example_srf_highcorr_OT_20240420_54.feather")

    df = srf_plot_data(
        filename=data_dir / "20240420_54_awake/01_spatial_receptive_field.h5",
        channel_number=30,
    )
    df.to_feather(OUTDIR / "example_srf_highcorr_Forebrain_20240420_54.feather")


def srf_plot_data(
    filename: str | pathlib.Path,
    channel_number: int,
):
    rec = Recording(POwlFile(filename))
    positions = rec.aggregate_stim_params(stim_position)
    trial_responses = rec.response_rates(
        channel_number=channel_number, stimulus_index=0
    )
    df = pd.DataFrame(
        [
            {
                "channel_number": channel_number,
                "azimuth": azi,
                "elevation": ele,
                "response_mean": resp.mean(),
                "response_std": resp.std(),
                "n_trials": resp.size,
            }
            for (azi, ele), resp in group_by_multiparam(
                trial_responses, positions
            ).items()
        ]
    )
    return df


def conincident_plot_data(
    filename: str | pathlib.Path,
    channel_number1: int,
    channel_number2: int,
):
    rec = Recording(POwlFile(filename))
    fixed_varying = fixed_var_stimuli(rec, stim_level)
    fixed_level = cast(float, fixed_varying["fixed_value"])
    varying_levels = np.array(
        rec.aggregate_stim_params(
            stim_level, stimulus_index=fixed_varying["varying_index"]
        )
    )
    trial_relative_levels = varying_levels - fixed_level
    df = pd.DataFrame(
        {
            "driver_level": fixed_level,
            "competitor_level": varying_levels,
            "relative_level": trial_relative_levels,
            "channel_unit1": channel_number1,
            "channel_unit2": channel_number2,
            "spiketrain_unit1": rec.stim_spiketrains(
                channel_number=channel_number1, ignore_onset=0.000
            ),
            "spiketrain_unit2": rec.stim_spiketrains(
                channel_number=channel_number2, ignore_onset=0.000
            ),
        }
    )
    return df


if __name__ == "__main__":
    main()
