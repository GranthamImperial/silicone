import itertools

import matplotlib.pyplot as plt
import numpy as np


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
