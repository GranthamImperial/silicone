import src.silicone.PlotCorrelationsBetweenGases as pltcor
from download_sr15_emissions import download_or_load_sr15
# Inputs to the code for PlotCorrelationsBetweenGases, freely modifiable
# ________________________________________________________

years_of_interest = [2030, 2050, 2100]
save_results = '../Output/'
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
# where do we get the data from?
SR15_SCENARIOS = "./sr15_scenarios.csv"
sr15_data = download_or_load_sr15(SR15_SCENARIOS)

pltcor.plot_emission_correlations(sr15_data, years_of_interest, save_results, plot_quantiles, quantiles_savename,
                                  quantile_boxes, quantile_decay_factor, smoothing_spline, model_colours,
                                  legend_fraction)
