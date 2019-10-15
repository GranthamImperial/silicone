import os.path

from click.testing import CliRunner

from silicone.cli import plot_correlations_between_gases


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
