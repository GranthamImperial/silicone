import os.path
import sys

import numpy as np
import pyam
import seaborn as sns
import pandas as pd

# Inputs to the code, to be read from a config file later

years_of_interest = [2020, 2030, 2050, 2100]
save_results = True

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

    # Initialise the tables to hold all parameters
    correlations_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])
    rank_corr_df = pd.DataFrame(index=df_gases.index, columns=[x_gas])
    for y_gas_ind in range(df_gases.count()[0]):
        y_gas = df_gases.index[y_gas_ind]
        y_units = df_gases.get('unit')[y_gas_ind]
        seaborn_df = sr15_data.filter(
            variable=[y_gas, x_gas],
            region="World",
            year=year_of_interest,
        ).pivot_table(
            ["year", "model", "scenario", "region"],
            ["variable"],
            aggfunc="mean"
        ).reset_index()

        # Data cleaning
        seaborn_df[y_gas].loc[seaborn_df[y_gas]==""] = np.nan
        seaborn_df[x_gas].loc[seaborn_df[x_gas]==""] = np.nan
        seaborn_df = seaborn_df.dropna()
        seaborn_df.loc[:, [y_gas, x_gas]] = seaborn_df[
            [y_gas, x_gas]
        ].astype(float)

        plotted_fig = sns.jointplot(
            x=x_gas,
            y=y_gas,
            data=seaborn_df,
            kind="reg"
        ).set_axis_labels(
            "Emissions of " + x_gas[10:] + " (" + x_units + ")",
            "Emissions of " + y_gas[10:] + " (" + y_units + ")"
        )
        if save_results:
            plotted_fig.savefig('../Output/Plot'+x_gas[10:]+'_'+y_gas[10:]+str(year_of_interest)+'.png')

        correlations_df.at[y_gas, x_gas] = seaborn_df.corr('pearson').loc[x_gas, y_gas]
        rank_corr_df.at[y_gas, x_gas] = seaborn_df.corr('spearman').loc[x_gas, y_gas]

    print(correlations_df)
    print(rank_corr_df)

    if save_results:
        correlations_df.to_csv('../Output/gasesCorrelation' + str(year_of_interest) + '.csv')
        rank_corr_df.to_csv('../Output/gasesRankCorr' + str(year_of_interest) + '.csv')
