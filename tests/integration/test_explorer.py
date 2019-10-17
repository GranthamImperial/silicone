import os.path

import pandas as pd
from click.testing import CliRunner

from silicone.cli import plot_emission_correlations_cli


def _make_input_file(df, dpath):
    input_dir = os.path.join(dpath, "input")
    if not os.path.isdir(input_dir):
        os.makedirs(input_dir)

    input_file = os.path.join(input_dir, "test_db.csv")
    df.to_csv(input_file)

    return input_file


def test_plot_emission_correlations(check_aggregate_df, tmpdir, caplog):
    runner = CliRunner(mix_stderr=False)

    input_file = _make_input_file(check_aggregate_df, tmpdir)

    years_of_interest = "2010"
    plot_quantiles = "0.2,0.5,0.8"
    output_dir = os.path.join(tmpdir, "output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    quantile_boxes = 3
    quantile_decay_factor = 1
    legend_fraction = 0.65

    # caplog can capture any logging calls (if we add any in future)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            plot_emission_correlations_cli,
            [
                input_file,
                "--output-dir",
                output_dir,
                "--years",
                years_of_interest,
                "--quantiles",
                plot_quantiles,
                "--quantile-boxes",
                quantile_boxes,
                "--quantile-decay-factor",
                quantile_decay_factor,
                "--model-colours",
                "--legend-fraction",
                legend_fraction,
            ],
        )

    quantiles_files = os.listdir(output_dir)
    assert quantiles_files[1].startswith("CO2_")

    png_files, csv_files = _get_png_and_csv_files(output_dir)

    CH4_file = "CO2_CH4_2010.csv"
    assert CH4_file in csv_files
    with open(os.path.join(output_dir, CH4_file)) as f:
        res_csv = pd.read_csv(f, delimiter=",")
        assert res_csv.iloc[1, 1] == 217

    assert len(csv_files) == 6
    assert len(png_files) == 4


def test_plot_emission_correlations_no_quantiles_or_model_colours(
    check_aggregate_df, tmpdir, caplog
):
    runner = CliRunner(mix_stderr=False)

    input_file = _make_input_file(check_aggregate_df, tmpdir)

    years_of_interest = "2010"
    output_dir = os.path.join(tmpdir, "output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    quantile_boxes = 3
    quantile_decay_factor = 1
    legend_fraction = 0.65

    # caplog can capture any logging calls (if we add any in future)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            plot_emission_correlations_cli,
            [
                input_file,
                "--output-dir",
                output_dir,
                "--years",
                years_of_interest,
                "--quantile-boxes",
                quantile_boxes,
                "--quantile-decay-factor",
                quantile_decay_factor,
                "--legend-fraction",
                legend_fraction,
            ],
        )

    png_files, csv_files = _get_png_and_csv_files(output_dir)
    assert len(csv_files) == 2
    assert len(png_files) == 4


def test_plot_emission_correlations_no_output(check_aggregate_df, tmpdir, caplog):
    runner = CliRunner(mix_stderr=False)

    input_file = _make_input_file(check_aggregate_df, tmpdir)

    years_of_interest = "2010"
    output_dir = os.path.join(tmpdir, "output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    quantile_boxes = 3
    quantile_decay_factor = 1
    legend_fraction = 0.65

    # caplog can capture any logging calls (if we add any in future)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            plot_emission_correlations_cli,
            [
                input_file,
                "--years",
                years_of_interest,
                "--quantile-boxes",
                quantile_boxes,
                "--quantile-decay-factor",
                quantile_decay_factor,
                "--legend-fraction",
                legend_fraction,
            ],
        )

    png_files, csv_files = _get_png_and_csv_files(output_dir)
    assert len(csv_files) == 0
    assert len(png_files) == 0


def test_plot_emission_correlations_no_co2_emms(check_aggregate_df, tmpdir, caplog):
    runner = CliRunner(mix_stderr=False)

    input_file = _make_input_file(
        check_aggregate_df.filter(variable="Primary Energy"), tmpdir
    )

    years_of_interest = "2010"
    output_dir = os.path.join(tmpdir, "output")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    quantile_boxes = 3
    quantile_decay_factor = 1
    legend_fraction = 0.65

    # caplog can capture any logging calls (if we add any in future)
    with caplog.at_level("DEBUG"):
        result = runner.invoke(
            plot_emission_correlations_cli,
            [
                input_file,
                "--output-dir",
                output_dir,
                "--years",
                years_of_interest,
                "--quantile-boxes",
                quantile_boxes,
                "--quantile-decay-factor",
                quantile_decay_factor,
                "--legend-fraction",
                legend_fraction,
            ],
        )

    assert result.exit_code  # make sure non-zero exit code given
    assert isinstance(result.exception, ValueError)
    assert result.exception.args == ("No Emissions|CO2 data",)

    png_files, csv_files = _get_png_and_csv_files(output_dir)
    assert len(csv_files) == 0
    assert len(png_files) == 0


def _get_png_and_csv_files(dir_to_search):
    # Detect file types
    quantiles_files = os.listdir(dir_to_search)

    png_files = [x for x in quantiles_files if x.endswith(".png")]
    csv_files = [x for x in quantiles_files if x.endswith(".csv")]

    return png_files, csv_files
