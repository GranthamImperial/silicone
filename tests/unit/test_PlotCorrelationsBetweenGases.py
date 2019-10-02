import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')
import src.silicone.PlotCorrelationsBetweenGases as PlotCorrelationsBetweenGases
from conftest import CHECK_AGG_DF



def test_PlotCorrelationsBetweenGases(check_aggregate_df):
    years_of_interest = [2010]
    save_results = False
    # if non-null, also plot these quantiles.
    plot_quantiles = [0.2, 0.5, 0.8]
    # if non-null, save data on the quantiles too
    quantiles_savename = '../../Output/TestQuantiles/'
    # How many boxes are used to fit the quantiles?
    quantile_boxes = 3
    # Should we extend the quantile boxes by an additional factor?
    quantile_decay_factor = 1
    # use a smoothing spline? If None, don't. Otherwise this is the smoothing factor, s, used in the spline model.
    smoothing_spline = None
    # Color different models different colours?
    model_colours = True
    # In the model-coloured version, how much does the figure need to be reduced by to leave room for the legend?
    legend_fraction = 0.65
    # ________________________________________________________

    PlotCorrelationsBetweenGases.plot_emission_correlations(check_aggregate_df, years_of_interest, save_results, plot_quantiles,
                                    quantiles_savename, quantile_boxes, quantile_decay_factor, smoothing_spline,
                                    model_colours, legend_fraction)
    quantiles_file = os.listdir(quantiles_savename)
    assert quantiles_file[1][0:4] == 'CO2_'

