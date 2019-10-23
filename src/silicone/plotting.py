"""Utility plotting functions"""
import itertools
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyam import IamDataFrame

from .database_crunchers import DatabaseCruncherRollingWindows


def plot_emission_correlations(
    emms_df,
    output_dir,
    years,
    quantiles,
    quantile_boxes,
    quantile_decay_factor,
    model_colours,
    legend_fraction,
):
    if quantiles is not None:
        cruncher = DatabaseCruncherRollingWindows(
            emms_df.filter(region="World", level=0, variable="Emissions|*")
        )

    # re-write to something like
    # loop over years
    # filter the data
    # derive the relationship
    # plot based on the filler, needs a new function, plot filler
    # plot raw data too
    for year_of_interest in years:
        # Obtain the list of gases to examine
        df_gases = (
            emms_df.filter(region="World", year=year_of_interest, level=1)
            .filter(variable="Emissions|*")
            .variables(True)
            .set_index("variable")
        )

        # We currently assume all correlations are with CO2
        x_gas = "Emissions|CO2"

        if x_gas not in df_gases.index:
            raise ValueError("No {} data".format(x_gas))

        # Check that the list has only one entry for each gas
        assert not any(df_gases.index.duplicated()), "Index contains duplicated entries"
        x_units = df_gases.loc[x_gas, "unit"]

        # Initialise the tables to hold all parameters between runs
        correlations_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])
        rank_corr_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])

        for y_gas_ind in range(df_gases.count()[0]):
            plt.close()
            y_gas = df_gases.index[y_gas_ind]
            y_units = df_gases.get("unit")[y_gas_ind]

            # Create the dataframe to plot
            seaborn_df = emms_df.filter(
                variable=[y_gas, x_gas], region="World", year=year_of_interest
            ).pivot_table(
                ["year", "model", "scenario", "region"], ["variable"], aggfunc="mean"
            )

            # Cleaning the data
            seaborn_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
            seaborn_df = seaborn_df.dropna().reset_index()
            seaborn_df.loc[:, [y_gas, x_gas]] = seaborn_df[[y_gas, x_gas]].astype(float)

            # Plot the results
            if model_colours:
                plot_multiple_models(
                    legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units
                )
            # if all plots are the same colour, we don't have to do all this work
            else:
                plot_emissions(seaborn_df, x_gas, y_gas, x_units, y_units)

            # Optionally calculate and plot quantiles
            if quantiles is not None:
                x_max = seaborn_df[x_gas].max()
                x_range = x_max - seaborn_df[x_gas].min()
                x_range = x_range if not np.equal(x_range, 0) else np.abs(0.1 * x_max)
                no_x_pts = 101
                x_pts = np.linspace(
                    seaborn_df[x_gas].min() - 0.1 * x_range,
                    seaborn_df[x_gas].max() + 0.1 * x_range,
                    no_x_pts,
                )

                tmp_df = (
                    cruncher._db.filter(variable=x_gas, year=year_of_interest)
                    .data.iloc[0, :]
                    .to_frame()
                    .T
                )
                tmp_df = pd.concat([tmp_df] * len(x_pts))
                tmp_df["value"] = x_pts
                tmp_df["scenario"] = [str(i) for i in range(no_x_pts)]
                tmp_df = IamDataFrame(tmp_df)

                smooth_quant_df = pd.DataFrame(columns=x_pts, index=quantiles)
                for quantile in quantiles:
                    filler = cruncher.derive_relationship(
                        y_gas,
                        [x_gas],
                        quantile=quantile,
                        nwindows=quantile_boxes,
                        decay_length_factor=quantile_decay_factor,
                    )

                    filled_points = filler(tmp_df).timeseries().values.squeeze()
                    smooth_quant_df.loc[quantile, :] = filled_points
                    plt.plot(
                        x_pts,
                        filled_points,
                        label=quantile if not model_colours else None,
                    )

                if not model_colours:
                    plt.legend()

                if output_dir is not None:
                    smooth_quant_df.to_csv(
                        os.path.join(
                            output_dir,
                            "{}_{}_{}.csv".format(
                                x_gas.replace("Emissions|", ""),
                                y_gas.replace("Emissions|", ""),
                                year_of_interest,
                            ),
                        )
                    )

            # Report the results
            if output_dir is not None:
                plt.savefig(
                    os.path.join(
                        output_dir,
                        "{}_{}_{}.png".format(x_gas[10:], y_gas[10:], year_of_interest),
                    )
                )

            correlations_df.at[y_gas, x_gas] = seaborn_df.corr("pearson").loc[
                x_gas, y_gas
            ]
            rank_corr_df.at[y_gas, x_gas] = seaborn_df.corr("spearman").loc[
                x_gas, y_gas
            ]
            print("Finished {} vs {} in {}".format(x_gas, y_gas, year_of_interest))

        print(correlations_df)
        print(rank_corr_df)
        if output_dir is not None:
            correlations_df.to_csv(
                os.path.join(
                    output_dir, "gases_correlation_{}.csv".format(year_of_interest)
                )
            )
            rank_corr_df.to_csv(
                os.path.join(
                    output_dir, "gases_rank_correlation_{}.csv".format(year_of_interest)
                )
            )


def plot_emissions(seaborn_df, x_gas, y_gas, x_units, y_units):
    colours_for_plot = "black"
    plt.scatter(x=x_gas, y=y_gas, label=colours_for_plot, data=seaborn_df, alpha=0.5)
    plt.xlabel("Emissions of {} ({})".format(x_gas[10:], x_units))
    plt.ylabel("Emissions of {} ({})".format(y_gas[10:], y_units))


def plot_multiple_models(legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units):
    ax = plt.subplot(111)
    all_models = list(seaborn_df["model"].unique())
    markers = itertools.cycle(["s", "o", "v", "<", ">", ","])
    for model in all_models:
        to_plot = np.where(seaborn_df["model"] == model)[0]
        if any(to_plot):
            plt.scatter(
                x=seaborn_df[x_gas].loc[to_plot],
                y=seaborn_df[y_gas].loc[to_plot],
                label=model,
                alpha=0.5,
                marker=next(markers),
            )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * legend_fraction, box.height])
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.xlabel("Emissions of {} ({})".format(x_gas[10:], x_units))
    plt.ylabel("Emissions of {} ({})".format(y_gas[10:], y_units))
