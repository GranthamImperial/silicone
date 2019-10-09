import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import silicone.utils
from silicone import utils

"""
Calculates the relationship between CO2 and other emissions. Prints to screen and, optionally, saves the correlations
and rank correlations between these emissions to the local 'Output' folder. Plots the relationships between these
emissions, optionally colouring different Optionally calculates the  

Parameters
----------
emissions_data : pyam.IamDataFrame 
    The emissions for each year 
years_of_interest : list 
    for which years should we explore the data?
save_results : String 
    If not None, the location in which all images and calculations except quantiles are saved to.  
plot_quantiles : list
    if None, do not calculate or quantiles, otherwise calculate and plot quantiles of the data on the graph. 
    Quantiles are calculated by a rolling filter. All parameters beginning 'quantile' are irrelevant if this is None. 
quantiles_savename : string
    if not None, the quantiles are saved to this location. 
quantile_boxes : int
    the number of points at which we evaluate the quantiles
quantile_decay_factor: float
    multiplicative factor that determines how rapidly the quantile filter falls to 0. 
    Default is one, meaning that data halfway between quantile evaluation points matters half as much as the 
    data at that point. Lower values means that the filter falls to 0 slower. 
    Formula is 1/(1+(distance/(box_width*decay_factor/2)^2) 
model_colours : boolean
    Should different models be distinguishable on the plots? If false, legend_fraction is ignored.  
legend_fraction : float
    In the model-coloured version, how much does the figure need to be reduced by to leave room for the legend?
"""

def plot_emission_correlations(emissions_data, years_of_interest, save_results, plot_quantiles, quantiles_savename,
                               quantile_boxes, quantile_decay_factor=1, model_colours=False, legend_fraction=1):

    for year_of_interest in years_of_interest:
        # Obtain the list of gases to examine
        df_gases = emissions_data.filter(
            region="World",
            year=year_of_interest,
            level=1
        ).filter(variable="Emissions|*").variables(True).set_index('variable')

        # We currently assume all correlations are with CO2
        x_gas = "Emissions|CO2"

        if x_gas not in df_gases.index:
            print('No emissions data')
            return None

        # Check that the list has only one entry for each gas
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
            seaborn_df = emissions_data.filter(
                variable=[y_gas, x_gas],
                region="World",
                year=year_of_interest,
            ).pivot_table(
                ["year", "model", "scenario", "region"],
                ["variable"],
                aggfunc="mean"
            )

            # Cleaning the data
            seaborn_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
            seaborn_df = seaborn_df.dropna().reset_index()
            seaborn_df.loc[:, [y_gas, x_gas]] = seaborn_df[
                [y_gas, x_gas]
            ].astype(float)

            # Plot the results
            if model_colours:
                plot_multiple_models(legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units)
            # if all plots are the same colour, we don't have to do all this work
            else:
                plot_emissions(seaborn_df, x_gas, y_gas, x_units, y_units)

            # Optionally calculate and plot quantiles
            if plot_quantiles is not None:
                smooth_quant_df = utils.rolling_window_find_quantiles(seaborn_df[x_gas], seaborn_df[y_gas],
                                                                      plot_quantiles, quantile_boxes,
                                                                      quantile_decay_factor)
                plt.plot(smooth_quant_df.index, smooth_quant_df)
                if not model_colours:
                    plt.legend(smooth_quant_df.keys())
                if quantiles_savename is not None:
                    silicone.utils.ensure_savepath(quantiles_savename)
                    smooth_quant_df.to_csv(quantiles_savename + x_gas[10:] + '_' + y_gas[10:] + '_' +
                                           str(year_of_interest) + '.csv')

            # Report the results
            if save_results is not None:
                silicone.utils.ensure_savepath(save_results)
                plt.savefig(save_results + 'Plot' + x_gas[10:] + '_' + y_gas[10:] + str(year_of_interest) + '.png')

            correlations_df.at[y_gas, x_gas] = seaborn_df.corr('pearson').loc[x_gas, y_gas]
            rank_corr_df.at[y_gas, x_gas] = seaborn_df.corr('spearman').loc[x_gas, y_gas]
            print('Finished ' + x_gas + ' vs ' + y_gas + ' in ' + str(year_of_interest))

        print(correlations_df)
        print(rank_corr_df)

        if save_results is not None:
            correlations_df.to_csv(save_results + 'gasesCorrelation' + str(year_of_interest) + '.csv')
            rank_corr_df.to_csv(save_results + 'gasesRankCorr' + str(year_of_interest) + '.csv')



def plot_emissions(seaborn_df, x_gas, y_gas, x_units, y_units):
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


def plot_multiple_models(legend_fraction, seaborn_df, x_gas, y_gas, x_units, y_units):
    fig = plt.figure(figsize=(16, 12))
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
    plt.xlabel("Emissions of " + x_gas[10:] + " (" + x_units + ")")
    plt.ylabel("Emissions of " + y_gas[10:] + " (" + y_units + ")")

