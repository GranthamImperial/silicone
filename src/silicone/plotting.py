"""Utility plotting functions"""
import itertools
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .stats import rolling_window_find_quantiles


def _plot_emission_correlations_quantile_rolling_windows(
    emms_df,
    output_dir,
    years,
    x_gas="Emissions|CO2",
    quantiles=None,
    quantile_boxes=20,
    quantile_decay_factor=1,
    model_colours=True,
    legend_fraction=0.65,
    region="World",
):
    """
    Calculates the relationship between different sorts of emissions at pre-specified
    times. Saves csv files of the correlation coefficients and the rank correlation
    coefficients between emissions at specified locations.

    Parameters
    ----------
    emms_df : :obj:`pyam.IamDataFrame`
        The database to search for correlations between named values

    output_dir : str
        The folder location to save the files.

    years : list[int]
        The years upon which to calculate correlations.

    x_gas : str
        The name of the gas to

    quantiles : list[float]
        If not none, the function will also calculate the quantiles specified by this
        list, using rolling windows as documented in rolling_window_find_quantiles. If
        none, the following four values are irrelevant.

    quantile_boxes : int
        The number of points at which quantiles should be evaluated. For details see
        rolling_window_find_quantiles documentation.

    quantile_decay_factor : float
        This determines how strong the local weighting is for the quantiles. 1 is the
        standard value. For details see rolling_window_find_quantiles documentation.

    model_colours : bool
        If true, the plot of quantiles will include a legend and differently coloured
        trends.

    legend_fraction : float
        The size of the legend, if plotted.
    """
    for year_of_interest in years:
        # Obtain the list of gases to examine
        df_gases = (
            emms_df.filter(region=region, year=year_of_interest, level=1)
            .filter(variable="Emissions|*")
            .variables(True)
            .set_index("variable")
        )

        if x_gas not in emms_df.variables().values:
            raise ValueError("No {} data".format(x_gas))

        # Check that the list has only one entry for each gas
        assert not any(df_gases.index.duplicated()), "Index contains duplicated entries"
        x_units = emms_df.filter(variable=x_gas)['unit'].iloc[0]

        # Initialise the tables to hold all parameters between runs
        correlations_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])
        rank_corr_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])

        for y_gas_ind in range(df_gases.count()[0]):
            plt.close()
            y_gas = df_gases.index[y_gas_ind]
            y_units = df_gases.get("unit")[y_gas_ind]

            # Create the dataframe to plot
            seaborn_df = emms_df.filter(
                variable=[y_gas, x_gas], region=region, year=year_of_interest
            ).pivot_table(
                ["year", "model", "scenario", "region"], ["variable"], aggfunc="mean"
            )

            # Cleaning the data
            seaborn_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
            seaborn_df = seaborn_df.dropna().reset_index()
            seaborn_df.loc[:, [y_gas, x_gas]] = seaborn_df[[y_gas, x_gas]].astype(float)

            # Plot the results
            if model_colours:
                _plot_multiple_models(
                    legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units
                )
            # if all plots are the same colour, we don't have to do all this work
            else:
                _plot_emissions(
                    legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units
                )

            # Optionally calculate and plot quantiles
            if quantiles is not None:
                quant_df = rolling_window_find_quantiles(
                    seaborn_df[x_gas],
                    seaborn_df[y_gas],
                    quantiles,
                    quantile_boxes,
                    quantile_decay_factor,
                )
                plt.plot(quant_df.index, quant_df)

                if not model_colours:
                    plt.legend(
                        quant_df.keys(), loc="center left", bbox_to_anchor=(1, 0.5)
                    )

                if output_dir is not None:
                    quant_df.to_csv(
                        os.path.join(
                            output_dir,
                            "{}_{}_{}.csv".format(
                                x_gas.split('|')[-1],
                                y_gas.split('|')[-1],
                                year_of_interest,
                            ),
                        )
                    )

            # Report the results
            if output_dir is not None:
                plt.savefig(
                    os.path.join(
                        output_dir,
                        "{}_{}_{}.png".format(x_gas.split('|')[-1], y_gas.split('|')[-1], year_of_interest),
                    )
                )

            correlations_df.at[y_gas, x_gas] = seaborn_df.corr("pearson").loc[
                x_gas, y_gas
            ]
            rank_corr_df.at[y_gas, x_gas] = seaborn_df.corr("spearman").loc[
                x_gas, y_gas
            ]
            print("Finished {} vs {} in {}".format(x_gas, y_gas, year_of_interest))

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


def _plot_emissions(legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units):
    ax = plt.subplot(111)
    colours_for_plot = "black"
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * legend_fraction, box.height])
    plt.scatter(x=x_gas, y=y_gas, color=colours_for_plot, data=seaborn_df, alpha=0.5)
    plt.xlabel("Emissions of {} ({})".format(x_gas.split('|')[-1], x_units))
    plt.ylabel("Emissions of {} ({})".format(y_gas.split('|')[-1], y_units))


def _plot_multiple_models(legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units):
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
    plt.xlabel("Emissions of {} ({})".format(x_gas.split('|')[-1], x_units))
    plt.ylabel("Emissions of {} ({})".format(y_gas.split('|')[-1], y_units))
