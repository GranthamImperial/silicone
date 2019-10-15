import os.path

import pandas as pd
from click.testing import CliRunner

from silicone.cli import plot_correlations_between_gases

# TODO:
# - test of what happens if you run with nothing to correlate

def test_plot_correlations_between_gases(check_aggregate_df, tmpdir, caplog):
    runner = CliRunner(mix_stderr=False)

    INPUT_DIR = os.path.join(tmpdir, "input")
    if not os.path.isdir(INPUT_DIR):
        os.makedirs(INPUT_DIR)

    INPUT_FILE = os.path.join(INPUT_DIR, "test_plot_correlations_between_gases_db.csv")
    check_aggregate_df.to_csv(INPUT_FILE)

    YEARS_OF_INTEREST = [2010]
    PLOT_QUANTILES = [0.2, 0.5, 0.8]
    OUTPUT_DIR = os.path.join(tmpdir, "output")
    if not os.path.isdir(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    QUANTILE_BOXES = 3
    QUANTILE_DECAY_FACTOR = 1
    MODEL_COLOURS = True
    LEGEND_FRACTION = 0.65

    # caplog can capture any logging calls (if we add any in future)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            plot_correlations_between_gases,
            [
                INPUT_FILE,
                OUTPUT_DIR,
                "--years",
                YEARS_OF_INTEREST,
                "--quantiles",
                PLOT_QUANTILES,
                "--quantile-boxes",
                QUANTILE_BOXES,
                "--quantile-decay-factor",
                QUANTILE_DECAY_FACTOR,
                "--model-colours",
                MODEL_COLOURS,
                "--legend-fraction",
                LEGEND_FRACTION,
            ],
        )

    quantiles_files = os.listdir(OUTPUT_DIR)
    assert quantiles_files[1].startswith('CO2_')

    png_files, csv_files = _get_png_and_csv_files(OUTPUT_DIR)

    CH4_file = 'CO2_CH4_2010.csv'
    assert CH4_file in csv_files
    with open(os.path.join(OUTPUT_DIR, CH4_file)) as f:
        res_csv = pd.read_csv(f, delimiter=',')
        assert res_csv.iloc[1, 1] == 217

    assert len(csv_files) == 4
    assert len(png_files) == 0


def _get_png_and_csv_files(dir_to_search):
    # Detect file types
    quantiles_files = os.listdir(dir_to_search)

    png_files = [x for x in quantiles_files if x.endswith(".png")]
    csv_files = [x for x in quantiles_files if x.endswith(".csv")]

    return png_files, csv_files
