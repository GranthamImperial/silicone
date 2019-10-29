"""
This script creates graphs and statistics for the relationships between emissions of the differences between SSP
models for a given scenario.
"""

from download_sr15_emissions import download_or_load_sr15
import silicone.PlotCorrelationsBetweenGases as pltcor

# Get the data
SR15_SCENARIOS = "./sr15_scenarios.csv"
sr15_data = download_or_load_sr15(SR15_SCENARIOS)

# _____Input values
# Indicate which scenarios are of interest
scenarios_of_interest = sr15_data.filter(scenario="SSP*").scenarios()
scenarios_of_interest = set([scenario[0:4] + "*" for scenario in scenarios_of_interest])

years_of_interest = [2030]
# Address of folder where info is saved to. We will append the scenario name.
save_results_stem = "../Output/Scenarios/"
# if non-null, also plot these quantiles.
plot_quantiles = None
# if non-null, save data on the quantiles too
quantiles_savename = "../Output/Scenarios/Quantiles/"
# How many boxes are used to fit the quantiles?
quantile_boxes = 5
# Should we extend the quantile boxes by an additional factor?
quantile_decay_factor = 0.7
# use a smoothing spline? If None, don't. Otherwise this is the smoothing factor, s, used in the spline model.
smoothing_spline = None
# Color different models different colours?
models_separate = True
# In the model-coloured version, how much does the figure need to be reduced by to leave room for the legend?
legend_fraction = 0.65
# ________________________________________________________
# ________________

for scenario in scenarios_of_interest:
    scenario_data = sr15_data.filter(scenario=scenario)

    save_results = save_results_stem + scenario[:-1]
    pltcor.plot_emission_correlations(
        scenario_data,
        years_of_interest,
        save_results,
        plot_quantiles,
        quantiles_savename,
        quantile_boxes,
        quantile_decay_factor,
        models_separate,
        legend_fraction,
    )
