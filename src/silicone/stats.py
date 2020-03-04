"""
Silicone's custom statistical operations.
"""
import numpy as np
import pandas as pd
import os


def rolling_window_find_quantiles(
    xs, ys, quantiles, nwindows=10, decay_length_factor=1
):
    """
    Perform quantile analysis in the y-direction for x-weighted data.

    Divides the x-axis into nwindows of equal length and weights data by how close they
    are to the center of these boxes. Then returns the quantiles of this weighted data.
    Quantiles are defined so that the values returned are always equal to a y-value in
    the data - there is no interpolation. Extremal points are given their full
    weighting, meaning this will not agree with the np.quantiles under uniform weighting
    (which effectively gives 0 weight to min and max values)

    The weighting of a point at :math:`x` for a window centered at :math:`x_0` is:

    .. math::
        w = \\frac{1}{1 + \\left (\\frac{x - x_0} {\\text{box_length} \\times \\text{decay_length_factor}} \\right )^2}

    Parameters
    ----------
    xs : np.ndarray, :obj:`pd.Series`
        The x co-ordinates to use in regression.

    xs : np.ndarray, :obj:`pd.Series`
        The x co-ordinates to use in regression.

    quantiles : list-like
        The quantiles to calculate in each window

    nwindows : positive int
        How many points to evaluate between x_max and x_min.

    decay_length_factor : float
        gives the distance over which the weighting of the values falls to 1/4,
        relative to half the distance between boxes. Defaults to 1. Formula is
        :math:`w = \\left ( 1 + \\left( \\frac{\\text{distance}}{\\text{box_length} \\times \\text{decay_length_factor}} \\right)^2 \\right)^{-1}`.

    Returns
    -------
        :obj:`pd.DataFrame`
        Quantile values at the window centres.
    """
    assert xs.size == ys.size
    if xs.size == 1:
        return pd.DataFrame(index=[xs[0]] * nwindows, columns=quantiles, data=ys[0])
    step = (max(xs) - min(xs)) / (nwindows + 1)
    decay_length = step / 2 * decay_length_factor
    # We re-form the arrays in case they were pandas series with integer labels that
    # would mess up the sorting.
    sort_order = np.argsort(ys)
    ys = ys[sort_order]
    xs = xs[sort_order]
    if max(xs) == min(xs):
        # We must prevent singularity behaviour if all the points have the same x.
        box_centers = np.array([xs[0]])
        decay_length = 1
    else:
        # We want to include the max x point, but not any point above it.
        # The 0.99 factor prevents rounding error inclusion.
        box_centers = np.arange(min(xs), max(xs) + step * 0.99, step)
    quantmatrix = pd.DataFrame(index=box_centers, columns=quantiles)
    for ind in range(box_centers.size):
        weights = 1.0 / (1.0 + ((xs - box_centers[ind]) / decay_length) ** 2)
        weights /= sum(weights)
        cumsum_weights = np.cumsum(weights)
        for i_quantile in range(quantiles.__len__()):
            quantmatrix.iloc[ind, i_quantile] = min(
                ys[cumsum_weights >= quantiles[i_quantile]]
            )
    return quantmatrix


def calc_all_correlations(emms_df, years, output_dir):
    """
   Saves csv files of the correlation coefficients and the rank correlation
    coefficients between emissions at specified locations.

    Parameters
    ----------
    emms_df : :obj:`pyam.IamDataFrame`
        The database to search for correlations between named values

    output_dir : str
        The folder location to save the files.

    years : list[int]
        The years upon which to calculate correlations.
    """
    assert len(emms_df.regions()) == 1, "Calculation is for only one region"
    region = emms_df.regions()[0]
    for year_of_interest in years:
        # Obtain the list of gases to examine
        df_gases = (
            emms_df.filter(year=year_of_interest, level=1)
            .filter(variable="Emissions|*")
            .append(
                emms_df.filter(year=year_of_interest, level=2)
                .filter(variable="Emissions|CO2*")
            )
            .variables(True)
            .set_index("variable")
        )
        # Initialise the tables to hold all parameters between runs
        correlations_df = pd.DataFrame(index=df_gases.index, columns=df_gases.index)
        rank_corr_df = pd.DataFrame(index=df_gases.index, columns=df_gases.index)
        # Check that the list has only one entry for each gas
        assert not any(df_gases.index.duplicated()), "Index contains duplicated entries"
        formatted_df = emms_df.filter(year=year_of_interest).pivot_table(
            ["year", "model", "scenario", "region"],
            ["variable"],
            aggfunc="mean"
        )
        formatted_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
        for x_gas_ind in range(df_gases.count()[0]):
            x_gas = df_gases.index[x_gas_ind]
            for y_gas_ind in range(df_gases.count()[0]):
                y_gas = df_gases.index[y_gas_ind]
                # Calculate the correlations. This requires removing NAs
                specific_df = formatted_df[
                    [y_gas, x_gas]
                ].astype(float).dropna()
                correlations_df.at[y_gas, x_gas] = specific_df.corr("pearson").loc[
                    x_gas, y_gas
                ]
                rank_corr_df.at[y_gas, x_gas] = specific_df.corr("spearman").loc[
                    x_gas, y_gas
                ]
        if output_dir is not None:
            correlations_df.to_csv(
                os.path.join(
                    output_dir, "gases_correlation_{}.csv".format(
                        year_of_interest
                    )
                )
            )
            rank_corr_df.to_csv(
                os.path.join(
                    output_dir, "gases_rank_correlation_{}.csv".format(
                        year_of_interest
                    )
                )
            )