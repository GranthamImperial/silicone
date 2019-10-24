"""Utility plotting functions"""
import itertools
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _plot_emission_correlations_cruncher_quantile_rolling_windows(
    emms_df,
    output_dir,
    years,
    quantiles,
    quantile_boxes,
    quantile_decay_factor,
    model_colours,
    legend_fraction,
):
    """

    """
    # TODO: split this function into smaller bits
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
                _plot_multiple_models(
                    legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units
                )
            # if all plots are the same colour, we don't have to do all this work
            else:
                _plot_emissions(seaborn_df, x_gas, y_gas, x_units, y_units)

            # Optionally calculate and plot quantiles
            if quantiles is not None:
                quant_df = _rolling_window_find_quantiles(
                    seaborn_df[x_gas],
                    seaborn_df[y_gas],
                    quantiles,
                    quantile_boxes,
                    quantile_decay_factor,
                )
                quant_df.plot(ax=plt.gca())

                if not model_colours:
                    plt.legend()

                if output_dir is not None:
                    quant_df.to_csv(
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


def _plot_emissions(seaborn_df, x_gas, y_gas, x_units, y_units):
    colours_for_plot = "black"
    plt.scatter(x=x_gas, y=y_gas, color=colours_for_plot, data=seaborn_df, alpha=0.5)
    plt.xlabel("Emissions of {} ({})".format(x_gas[10:], x_units))
    plt.ylabel("Emissions of {} ({})".format(y_gas[10:], y_units))


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
    plt.xlabel("Emissions of {} ({})".format(x_gas[10:], x_units))
    plt.ylabel("Emissions of {} ({})".format(y_gas[10:], y_units))


def _rolling_window_find_quantiles(xs, ys, quantiles, nboxes=10, decay_length_factor=1):
    # TODO: move this into DatabaseCruncherQuantileRollingWindows
    assert xs.size == ys.size
    if xs.size == 1:
        return pd.DataFrame(index=[xs[0]] * nboxes, columns=quantiles, data=ys[0])
    step = (max(xs) - min(xs)) / (nboxes + 1)
    decay_length = step / 2 * decay_length_factor
    # We re-form the arrays in case they were pandas series with integer labels that would mess up the sorting.
    xs = np.array(xs)
    ys = np.array(ys)
    sort_order = np.argsort(ys)
    ys = ys[sort_order]
    xs = xs[sort_order]
    if max(xs) == min(xs):
        # We must prevent singularity behaviour if all the points are at the same x value.
        box_centers = np.array([xs[0]])
        decay_length = 1
    else:
        # We want to include the max x point, but not any point above it.
        # The 0.99 factor prevents rounding error inclusion.
        box_centers = np.arange(min(xs), max(xs) + step * 0.99, step)
    quantmatrix = pd.DataFrame(index=box_centers, columns=quantiles)
    for ind in range(box_centers.size):
        weights = 1.0 / (1.0 + ((xs - box_centers[ind]) / decay_length) ** 2)
        weights /= sum(weights)
        # We want to calculate the weights at the midpoint of step corresponding to the y-value.
        cumsum_weights = np.cumsum(weights)
        for i_quantile in range(quantiles.__len__()):
            quantmatrix.iloc[ind, i_quantile] = min(
                ys[cumsum_weights >= quantiles[i_quantile]]
            )
    return quantmatrix
