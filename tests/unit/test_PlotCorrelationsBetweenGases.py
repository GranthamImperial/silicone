import sys
import os
myPath = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, myPath + '/../../')
import src.silicone.PlotCorrelationsBetweenGases as PlotCorrelationsBetweenGases
import pandas as pd
import re

def test_PlotCorrelationsBetweenGases(check_aggregate_df):
    # Assumptions:
    years_of_interest = [2010]
    # if non-null, also plot these quantiles.
    plot_quantiles = [0.2, 0.5, 0.8]
    # if non-null, save data on the quantiles too
    saveplace = '../../Output/Test/'
    # How many boxes are used to fit the quantiles?
    quantile_boxes = 3
    # Should we extend the quantile boxes by an additional factor?
    quantile_decay_factor = 1
    # Color different models different colours?
    model_colours = True
    # In the model-coloured version, how much does the figure need to be reduced by to leave room for the legend?
    legend_fraction = 0.65
    # ________________________________________________________

    # Clean the output folder before we start.
    if not os.path.isdir(saveplace):
        os.makedirs(saveplace)
    output_files = os.listdir(saveplace)
    for output_file in output_files:
        if output_file[-9:] != 'gitignore':
            os.remove(saveplace + output_file)
    output_files = os.listdir(saveplace)
    initial_files = len(output_files)

    # Run without anything to correlate, saving output (which should be nothing) to quantiles output folder
    PlotCorrelationsBetweenGases.plot_emission_correlations(check_aggregate_df.filter(variable='Primary Energy'),
                                    years_of_interest, saveplace, plot_quantiles, saveplace,
                                    quantile_boxes, quantile_decay_factor, model_colours,
                                    legend_fraction)
    # Nothing should be created by this
    assert output_files == os.listdir(saveplace)

    # Run the first set of parameters
    PlotCorrelationsBetweenGases.plot_emission_correlations(check_aggregate_df, years_of_interest, None,
                                    plot_quantiles, saveplace, quantile_boxes, quantile_decay_factor,
                                    model_colours, legend_fraction)
    quantiles_files = os.listdir(saveplace)
    assert quantiles_files[1][0:4] == 'CO2_'

    png_files, csv_files = png_and_csv_files(saveplace)
    with open(saveplace + csv_files[2]) as csv_file:
        csv_reader = pd.read_csv(csv_file, delimiter=',')
        assert csv_reader.iloc[1, 1] == 217
    count_and_delete_files(csv_files, saveplace, 4)
    assert len(png_files) == 0

    # Rerun the code with slightly different parameters to explore more code options.
    PlotCorrelationsBetweenGases.plot_emission_correlations(check_aggregate_df, years_of_interest, saveplace,
                                    plot_quantiles, saveplace, quantile_boxes, quantile_decay_factor,
                                    model_colours, legend_fraction)
    png_files, csv_files = png_and_csv_files(saveplace)
    with open(saveplace + csv_files[2]) as csv_file:
        csv_reader = pd.read_csv(csv_file, delimiter=',')
        assert csv_reader.iloc[1, 1] == 217

    count_and_delete_files(png_files, saveplace, 4)
    count_and_delete_files(csv_files, saveplace, 6)

    # Rerun the code with slightly different parameters to explore more code options.
    plot_quantiles = None
    save_results = '../../Output/Test/'
    model_colours = False
    PlotCorrelationsBetweenGases.plot_emission_correlations(check_aggregate_df, years_of_interest, save_results,
                                    plot_quantiles, saveplace, quantile_boxes, quantile_decay_factor,
                                    model_colours, legend_fraction)
    png_files, csv_files = png_and_csv_files(saveplace)
    count_and_delete_files(csv_files, saveplace, 2)
    count_and_delete_files(png_files, saveplace, 4)
    output_files = os.listdir(saveplace)
    # check no unexpected files created
    assert len(output_files) == initial_files


def count_and_delete_files(files, file_root, length_of_files):
    assert len(files) == length_of_files
    for files in files:
        os.remove(file_root + files)

def png_and_csv_files(saveplace):
    # Detect file types
    regex_match_png = re.compile(".*" + ".png")
    regex_match_csv = re.compile(".*" + ".csv")
    quantiles_files = os.listdir(saveplace)
    png_files = [x for x in quantiles_files if regex_match_png.match(x)]
    csv_files = [x for x in quantiles_files if regex_match_csv.match(x)]
    return png_files, csv_files