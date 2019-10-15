import itertools
import os.path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import click
import pyam


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "emissions_data", type=click.Path(exists=True, readable=True, resolve_path=True)
)
@click.option(
    "--output-dir",
    default=None,
    show_default=False,
    type=click.Path(writable=True, resolve_path=True),
    help="Directory in which to save output",
)
@click.option(
    "--years",
    default="2010, 2015",
    type=click.STRING,
    show_default=True,
    help="Years in which to explore correlations (comma-separated)",
)
@click.option(
    "--quantiles",
    default=None,
    type=click.STRING,
    show_default=False,
    help="Quantiles to calculate (comma-separated)",
)
@click.option(
    "--quantile-boxes",
    default=3,
    type=click.INT,
    show_default=True,
    help="Number of boxes (along the x-axis) to use when calculating quantiles",
)
@click.option(
    "--quantile-decay-factor",
    default=1,
    type=click.FLOAT,
    show_default=True,
    help="Multiplicative factor that determines how rapidly the quantile filter falls to 0",
)
@click.option(
    "--model-colours/--no-model-colours",
    help="Make different models distinguishable on the plots? If ``False``, ``legend_fraction`` is ignored.",
    default=False,
    show_default=True,
)
@click.option(
    "--legend-fraction",
    default=1,
    type=click.FLOAT,
    show_default=True,
    help="In the model-coloured version, how much does the figure need to be reduced by to leave room for the legend?",
)
def plot_emission_correlations(
    emissions_data,
    output_dir,
    years,
    quantiles,
    quantile_boxes,
    quantile_decay_factor,
    model_colours,
    legend_fraction,
):
    years = [int(y.strip()) for y in years.split(",")]
    if quantiles is not None:
        quantiles = [float(q.strip()) for q in quantiles.split(",")]

    emissions_data = pyam.IamDataFrame(emissions_data)
    for year_of_interest in years:
        # Obtain the list of gases to examine
        df_gases = (
            emissions_data.filter(region="World", year=year_of_interest, level=1)
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
            seaborn_df = emissions_data.filter(
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
                smooth_quant_df = rolling_window_find_quantiles(
                    seaborn_df[x_gas],
                    seaborn_df[y_gas],
                    quantiles,
                    quantile_boxes,
                    quantile_decay_factor,
                )
                plt.plot(smooth_quant_df.index, smooth_quant_df)
                if not model_colours:
                    plt.legend(smooth_quant_df.keys())

                if output_dir is not None:
                    smooth_quant_df.to_csv(
                        os.path.join(
                            output_dir,
                            "{}_{}_{}.csv".format(x_gas[10:], y_gas[10:], year_of_interest),
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
    fig = plt.figure(figsize=(16, 12))
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


def rolling_window_find_quantiles(xs, ys, quantiles, nboxes=10, decay_length_factor=1):
    assert xs.size == ys.size
    if xs.size == 1:
        return pd.DataFrame(index=[xs[0]] * nboxes, columns=quantiles, data=ys[0])
    step = (max(xs) - min(xs)) / (nboxes + 1)
    decay_length = step / 2 * decay_length_factor
    # We re-form the arrays in case they were pandas series with integer labels that
    # would mess up the sorting.
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
