import os.path
import sys

import numpy as np
import pyam
import pandas as pd
import matplotlib.pyplot as plt
import scipy.interpolate as sci
import itertools

from src.silicone import utils

# Inputs to the code, freely modifiable
# ________________________________________________________
years_of_interest = [2030, 2050, 2100]
save_results = True
# if non-null, also plot these quantiles.
plot_quantiles = [0.2, 0.33, 0.5, 0.67, 0.8]
# if non-null, save data on the quantiles too
quantiles_savename = '../Output/Quantiles/'
# How many boxes are used to fit the quantiles?
quantile_boxes = 15
# Should we extend the quantile boxes by an additional factor?
quantile_decay_factor = 0.7
# use a smoothing spline? If None, don't. Otherwise this is the smoothing factor, s, used in the spline model.
smoothing_spline = None
# Color different models different colours?
model_colours = True
# In the model-coloured version, how much does the figure need to be reduced by to leave room for the legend?
legend_fraction = 0.65
# ________________________________________________________

# Get the data
SR15_SCENARIOS = "./sr15_scenarios.csv"
if not os.path.isfile(SR15_SCENARIOS):
    sys.path.append('./../scripts/')
    from download_sr15_emissions import get_sr15_scenarios
    get_sr15_scenarios(SR15_SCENARIOS)
sr15_data = pyam.IamDataFrame(SR15_SCENARIOS)

for year_of_interest in years_of_interest:
    # Obtain the data as required
    df_gases = sr15_data.filter(
        region="World",
        year=year_of_interest,
        level=1
    ).variables('Emissions|*').set_index('variable')

    x_gas = "Emissions|CO2"

    # Check that the list has only one entry for carbon
    assert not any(df_gases.index.duplicated()), "Index contains duplicated entries"
    x_units = df_gases.loc[x_gas, 'unit']

    # Initialise the tables to hold all parameters between runs
    correlations_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])
    rank_corr_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])
    for y_gas_ind in range(df_gases.count()[0]):
        plt.close()
        y_gas = df_gases.index[y_gas_ind]
        y_units = df_gases.get('unit')[y_gas_ind]

        # Create the dataframe to plot
        seaborn_df = sr15_data.filter(
            variable=[y_gas, x_gas],
            region="World",
            year=year_of_interest,
        ).pivot_table(
            ["year", "model", "scenario", "region"],
            ["variable"],
            aggfunc="mean"
        )

        # Cleaning the data
        seaborn_df[y_gas].loc[seaborn_df[y_gas]==""] = np.nan
        seaborn_df[x_gas].loc[seaborn_df[x_gas]==""] = np.nan
        seaborn_df = seaborn_df.dropna().reset_index()
        seaborn_df.loc[:, [y_gas, x_gas]] = seaborn_df[
            [y_gas, x_gas]
        ].astype(float)

        # Plot the results
        if model_colours:
            fig = plt.figure()
            ax = plt.subplot(111)
            all_models = list(seaborn_df['model'].unique())
            markers = itertools.cycle(["s", "o", "v", "<", ">", ","])
            for model in all_models:
                to_plot = np.where(seaborn_df['model'] == model)[0]
                if any(to_plot):
                    plt.scatter(
                        x=seaborn_df[x_gas].loc[to_plot],
                        y=seaborn_df[y_gas].loc[to_plot],
                        label=model,
                        alpha=0.5,
                        marker=next(markers)
                    )
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * legend_fraction, box.height])
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # if all plots are the same colour, we don't have to do all this work
        else:
            colours_for_plot = 'black'
            plt.scatter(
                x=x_gas,
                y=y_gas,
                label=colours_for_plot,
                data=seaborn_df,
                alpha=0.5
            )
        plt.xlabel("Emissions of " + x_gas[10:] + " (" + x_units + ")")
        plt.ylabel("Emissions of " + y_gas[10:] + " (" + y_units + ")")

        # Optionally calculate and plot quantiles
        if plot_quantiles is not None:
            smooth_quant_df = utils.rolling_window_find_quantiles(seaborn_df[x_gas], seaborn_df[y_gas],
                                                                  plot_quantiles, quantile_boxes, quantile_decay_factor)
            if smoothing_spline is not None:
                for col in smooth_quant_df:
                    manyx = np.arange(min(smooth_quant_df.index), max(smooth_quant_df.index),
                                      (max(smooth_quant_df.index) - min(smooth_quant_df.index)) / 100)
                    spline = sci.UnivariateSpline(smooth_quant_df.index, smooth_quant_df[col], s=smoothing_spline)
                    plt.plot(manyx, spline(manyx))
            else:
                plt.plot(smooth_quant_df.index, smooth_quant_df)
            if not model_colours:
                plt.legend(smooth_quant_df.keys())
            if quantiles_savename is not None:
                smooth_quant_df.to_csv(quantiles_savename + x_gas[10:] + '_' + y_gas[10:] + '_' +
                                       str(year_of_interest) + '.csv')

        # Report the results
        if save_results:
            plt.savefig('../Output/Plot'+x_gas[10:]+'_'+y_gas[10:]+str(year_of_interest)+'.png')

        correlations_df.at[y_gas, x_gas] = seaborn_df.corr('pearson').loc[x_gas, y_gas]
        rank_corr_df.at[y_gas, x_gas] = seaborn_df.corr('spearman').loc[x_gas, y_gas]
        print('Finished ' + x_gas + ' vs ' + y_gas + ' in ' + str(year_of_interest))

    print(correlations_df)
    print(rank_corr_df)

    if save_results:
        correlations_df.to_csv('../Output/gasesCorrelation' + str(year_of_interest) + '.csv')
        rank_corr_df.to_csv('../Output/gasesRankCorr' + str(year_of_interest) + '.csv')
