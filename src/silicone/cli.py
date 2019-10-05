"""Command line interface"""
import click
import pyam

from .plotting import _plot_emission_correlations_cruncher_quantile_rolling_windows


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "emissions_data", type=click.Path(exists=True, readable=True, resolve_path=True)
)
@click.option(
    "--output-dir",
    default=None,
    show_default=False,
    type=click.Path(writable=True, resolve_path=True),
    help="Directory in which to save output",
)
@click.option(
    "--years",
    default="2010, 2015",
    type=click.STRING,
    show_default=True,
    help="Years in which to explore correlations (comma-separated)",
)
@click.option(
    "--quantiles",
    default=None,
    type=click.STRING,
    show_default=False,
    help="Quantiles to calculate (comma-separated)",
)
@click.option(
    "--quantile-boxes",
    default=3,
    type=click.INT,
    show_default=True,
    help="Number of boxes (along the x-axis) to use when calculating quantiles",
)
@click.option(
    "--quantile-decay-factor",
    default=1,
    type=click.FLOAT,
    show_default=True,
    help="Multiplicative factor that determines how rapidly the quantile filter falls to 0",
)
@click.option(
    "--model-colours/--no-model-colours",
    help="Make different models distinguishable on the plots? If ``False``, ``legend_fraction`` is ignored.",
    default=False,
    show_default=True,
)
@click.option(
    "--legend-fraction",
    default=1,
    type=click.FLOAT,
    show_default=True,
    help="In the model-coloured version, how much does the figure need to be reduced by to leave room for the legend?",
)
def plot_emission_correlations_cruncher_quantile_rolling_windows_cli(
    emissions_data,
    output_dir,
    years,
    quantiles,
    quantile_boxes,
    quantile_decay_factor,
    model_colours,
    legend_fraction,
):
    r"""
    Plot correlations between emisssions timeseries in ``emissions_data``.

    ``emissions_data`` is the file from which to load the emissions timeseries.
    """
    years = [int(y.strip()) for y in years.split(",")]
    if quantiles is not None:
        quantiles = [float(q.strip()) for q in quantiles.split(",")]

    emissions_data = pyam.IamDataFrame(emissions_data)

    _plot_emission_correlations_cruncher_quantile_rolling_windows(
        emissions_data,
        output_dir,
        years,
        quantiles,
        quantile_boxes,
        quantile_decay_factor,
        model_colours,
        legend_fraction,
    )


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument(
    "scenarios_raw", type=click.Path(exists=True, readable=True, resolve_path=True)
)
@click.argument(
    "scenarios_db", type=click.Path(exists=True, readable=True, resolve_path=True)
)
@click.argument("scenarios_output", type=click.Path(writable=True, resolve_path=True))
@click.option(
    "--config",
    default=None,
    type=click.Path(readable=True, resolve_path=True),
    show_default=True,
    help="Configuration to use when infilling (this should be a yaml file).",
)
def infill(scenarios_raw, scenarios_db, scenarios_output, config):
    r"""
    In fill data in ``scenarios_raw`` using information from ``scenarios_db``.

    The output is written in ``scenarios_output``.

    The configuration to use for the run is read from ``config``. If ``config`` is not
    supplied, the default config will be used. These can be found in
    ``silicone.cli.default_infill_config``.
    """
    raise NotImplementedError
