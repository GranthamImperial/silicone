"""
Silicone's custom statistical operations.
"""
import os

import numpy as np
import pandas as pd
import scipy.interpolate
import scipy.stats


def rolling_window_find_quantiles(
    xs, ys, quantiles, nwindows=11, decay_length_factor=1
):
    """
    Perform quantile analysis in the y-direction for x-weighted data.

    Divides the x-axis into nwindows of equal length and weights data by how close they
    are to the center of these windows. Then returns the quantiles of this weighted
    data. Quantiles are defined so that the values returned are always equal to a y-
    value in the data - there is no interpolation. Extremal points are given their full
    weighting, meaning this will not agree with the np.quantiles under uniform weighting
    (which effectively gives 0 weight to min and max values).

    The weighting of a point at :math:`x` for a window centered at :math:`x_0` is:

    .. math::

        w = \\frac{1}{1 + \\left (\\frac{x - x_0}{l_{window}} \\times f_{dl} \\right)^2}

    for :math:`l_{window}` the window width (range of values divided by nwindows -1) and
    :math:`f_{dl}` the decay_length_factor.

    Parameters
    ----------
    xs : np.ndarray, :obj:`pd.Series`
        The x co-ordinates to use in the regression.

    ys : np.ndarray, :obj:`pd.Series`
        The y co-ordinates to use in the regression.

    quantiles : list-like
        The quantiles to calculate in each window

    nwindows : int
        How many points to evaluate between x_max and x_min. Must be > 1.

    decay_length_factor : float
        gives the distance over which the weighting of the values falls to 1/4,
        relative to half the distance between window centres. Defaults to 1.

    Returns
    -------
    :obj:`pd.DataFrame`
        Quantile values at the window centres.

    Raises
    ------
    AssertionError
        ``xs`` and ``ys`` don't have the same shape
    """
    if xs.shape != ys.shape:
        raise AssertionError("`xs` and `ys` must be the same shape")

    if isinstance(quantiles, (float, np.float64)):
        quantiles = [quantiles]

    # min(xs) == max(xs) cannot be accessed via QRW cruncher, as a short-circuit appears
    # earlier in the code.
    if np.equal(max(xs), min(xs)):
        # We must prevent singularity behaviour if all the points have the same x.
        window_centers = np.array([xs[0]])
        decay_length = 1
        if np.equal(max(ys), min(ys)):
            return pd.DataFrame(index=window_centers, columns=quantiles, data=ys[0])

    else:
        # We want to include the max x point, but not any point above it.
        # The 0.99 factor prevents rounding error inclusion.
        step = (max(xs) - min(xs)) / (nwindows - 1)
        decay_length = step / 2 * decay_length_factor
        window_centers = np.arange(min(xs), max(xs) + step * 0.99, step)

    ys, xs = map(np.array, zip(*sorted(zip(ys, xs))))

    results = pd.DataFrame(index=window_centers, columns=quantiles)
    results.columns.name = "window_centers"

    for window_center in window_centers:
        weights = 1.0 / (1.0 + ((xs - window_center) / decay_length) ** 2)
        weights /= sum(weights)

        # We want to calculate the weights at the midpoint of step
        # corresponding to the y-value.
        cumsum_weights = np.cumsum(weights) - 0.5 * weights
        results.loc[window_center, quantiles] = scipy.interpolate.interp1d(
            cumsum_weights,
            ys,
            bounds_error=False,
            fill_value=(ys[0], ys[-1]),
            assume_sorted=True,
        )(quantiles)

    return results


def calc_quantiles_of_data(
    distribution, points_to_quant, smoothing=None, weighting=None, to_quantile=True
):
    """
    Calculates the quantiles of points_to_quant in the distribution of values described
    by distribution. Optionally treats points_to_quant as quantiles and returns the
    values that would lead to them instead.

    Parameters
    ----------
    distribution : pd.Series
        The distribution of values.
    points_to_quant : pd.Series
        The points which we want find: if ``to_quantile`` is ``True`` (default) these are the
        values which we will compare to the distribution, if ``False``, these are the
        quantiles which we want to find.
    smoothing : float or string
        By default, no smoothing is done on the distribution. If a value is
        provided, it is fed into :func:`scipy.stats.gaussian_kde` - see full
        documentation there. In short, if a float is input, we fit a Gaussian kernel
        density estimator with that width to the points. If a string is used, it
        must be either "scott" or "silverman", after those two methods of
        determining the best kernel bandwidth.
    weighting : None or Series
        If a series, must have the same indices as distribution, giving the
        relative weights of each point.
    to_quantile : Bool
        If ``True``, we return the quantiles of the data in points_to_quant. If ``False``, we
        instead treat points_to_quant as the quantiles themselves (they must all be 0-1)
        and return the values in distribution that occur at these quantiles.

    Returns
    -------
    :obj:`np.ndarray`
        An array with one row and a column for each entry in points_to_quant, containing the
        quantiles of these points in order. Or, if to_quantile is ``False``, containing the
        values corresponding to the quantiles points_to_quant.
    """
    if not to_quantile:
        assert all(
            [i <= 1 and i >= 0 for i in points_to_quant]
        ), "Quantiles are not defined outside the range 0-1"
    if weighting is None:
        weighting = pd.Series(np.ones_like(distribution), index=distribution.index)
    else:

        assert isinstance(
            weighting, pd.Series
        ), "The weighting variable should be a Series"
        assert len(weighting) == len(distribution), (
            "There must be the same number of weights as entries in the database. "
            "We have len {} in weights and len {} in distribution.".format(
                len(weighting), len(distribution)
            )
        )
        assert np.array_equal(
            weighting.index.sort_values(), distribution.index.sort_values()
        ), (
            "The entries in the weighting Series are not clearly aligned with the "
            "distribution Series. We have \n {} \n in the weighting and \n {} \n in "
            "the distribution ".format(weighting.index, distribution.index)
        )

    if any(np.isnan(distribution)):
        distribution = distribution[~np.isnan(distribution)]
        weighting = weighting[distribution.index]
    len_lead_not_nan = len(distribution)
    if len_lead_not_nan == 1:
        # If there is only a single value then quantile is not defined
        if to_quantile:
            return np.nan * np.zeros_like(points_to_quant)
        else:
            # all quantiles give the same point
            return distribution[0]
    if len_lead_not_nan == 0:
        raise ValueError("No valid data entered to establish the quantiles.")
    if smoothing:
        smooth_points = scipy.stats.gaussian_kde(
            distribution, smoothing, weights=weighting
        )
        minx = min(distribution)
        maxx = max(distribution)
        tail = (maxx - minx) / 2
        # tile the probability space with 10000 points between the tail ends
        xpts = np.linspace(minx - tail, maxx + tail, 10000)
        smooth_dist = np.cumsum(smooth_points(xpts)) * (xpts[1] - xpts[0])
        # But we only want to return values between the extrema
        xkeep = (xpts >= minx) & (xpts <= maxx)
        xpts = xpts[xkeep]
        smooth_dist = smooth_dist[xkeep]

        if to_quantile:
            return scipy.interpolate.interp1d(
                xpts, smooth_dist, bounds_error=False, fill_value=(0, 1)
            )(points_to_quant)
        else:
            return scipy.interpolate.interp1d(
                smooth_dist, xpts, bounds_error=False, fill_value=(minx, maxx)
            )(points_to_quant)

    dist_sort_ind = np.argsort(distribution)
    distribution = distribution[dist_sort_ind.index[dist_sort_ind]]
    weighting = weighting[dist_sort_ind.index[dist_sort_ind]]
    quant_of_lead_vals = (np.cumsum(weighting) - weighting) / (
        sum(weighting) - weighting
    )
    if any(quant_of_lead_vals > 1) or any(quant_of_lead_vals < 0):
        raise ValueError("Impossible quantiles!")
    if to_quantile:
        return scipy.interpolate.interp1d(
            distribution, quant_of_lead_vals, bounds_error=False, fill_value=(0, 1)
        )(points_to_quant)
    else:
        return scipy.interpolate.interp1d(quant_of_lead_vals, distribution)(
            points_to_quant
        )


def calc_all_emissions_correlations(emms_df, years, output_dir):
    """
    Save csv files of the correlation coefficients and the rank correlation
    coefficients between emissions at specified times.

    This function includes all undivided emissions (i.e. results recorded as
    `Emissions|X`) and CO2 emissions split once (i.e. `Emissions|CO2|X`). It does not
    include Kyoto gases. It will also save the average absolute value of the
    coefficients.

    Parameters
    ----------
    emms_df : :obj:`pyam.IamDataFrame`
        The database to search for correlations between named values

    output_dir : str
        The folder location to save the files.

    years : list[int]
        The years upon which to calculate correlations.

    Files created
    -------------
    "variable_counts.csv" : the number of scenario/model pairs where the emissions
    data occurs.

    "gases_correlation_{year}.csv" : The Pearson's correlation between gases emissions
     in a given year.

    "gases_rank_correlation_{year}.csv" : The Spearman's rank correlation between
    gases in a given year

    "time_av_absolute_correlation_{}_to_{}.csv" : The magnitude of the Pearson's
    correlation between emissions, averaged over the years requested.

    "time_av_absolute_rank_correlation_{}_to_{}.csv" : The magnitude of the Spearman's
     rank correlation between emissions, averaged over the years requested.

    "time_variance_rank_correlation_{}_to_{}.csv" : The variance over time in the rank
     correlation values above.
    """
    assert len(emms_df.region) == 1, "Calculation is for only one region"
    # Obtain the list of gases to examine
    df_gases = (
        emms_df.filter(level=1)
        .filter(variable="Emissions|*")
        .filter(variable="Emissions|Kyoto*", keep=False)
        .append(emms_df.filter(level=2).filter(variable="Emissions|CO2*"))[
            ["variable", "unit"]
        ]
        .drop_duplicates()
        .set_index("variable")
    )
    all_correlations_df = pd.DataFrame(
        index=df_gases.index, columns=df_gases.index, data=0
    )
    all_rank_corr_df = pd.DataFrame(
        index=df_gases.index, columns=df_gases.index, data=0
    )
    all_rank_corr_var_df = pd.DataFrame(
        index=df_gases.index, columns=df_gases.index, data=0
    )

    # Calculate the total amount of data
    var_count_file = "variable_counts.csv"
    var_count = pd.Series(index=df_gases.index, dtype=int)
    for var in df_gases.index:
        var_db = emms_df.filter(variable=var)
        var_count[var] = len(var_db.timeseries())
    var_save_loc = os.path.join(output_dir, var_count_file)
    var_count.to_csv(var_save_loc)
    print("Counted the number of each variable and saved to ".format(var_save_loc))

    for year_of_interest in years:
        # Initialise the tables to hold all parameters between runs
        correlations_df = pd.DataFrame(index=df_gases.index, columns=df_gases.index)
        rank_corr_df = pd.DataFrame(index=df_gases.index, columns=df_gases.index)
        # Check that the list has only one entry for each gas
        assert not any(df_gases.index.duplicated()), "Index contains duplicated entries"
        formatted_df = emms_df.filter(
            variable=df_gases.index, year=year_of_interest
        ).pivot_table(
            ["year", "model", "scenario", "region"], ["variable"], aggfunc="mean"
        )
        formatted_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
        for x_gas_ind in range(df_gases.count()[0]):
            x_gas = df_gases.index[x_gas_ind]
            for y_gas_ind in range(x_gas_ind + 1, df_gases.count()[0]):
                y_gas = df_gases.index[y_gas_ind]
                # Calculate the correlations. This requires removing NAs
                correlations_df.at[y_gas, x_gas] = formatted_df.corr("pearson").loc[
                    x_gas, y_gas
                ]
                rank_corr_df.at[y_gas, x_gas] = formatted_df.corr("spearman").loc[
                    x_gas, y_gas
                ]
                all_correlations_df.loc[y_gas, x_gas] = all_correlations_df.at[
                    y_gas, x_gas
                ] + abs(correlations_df.loc[y_gas, x_gas]) / len(years)
                all_rank_corr_df.loc[y_gas, x_gas] = all_rank_corr_df.at[
                    y_gas, x_gas
                ] + abs(rank_corr_df.at[y_gas, x_gas]) / len(years)
                all_rank_corr_var_df.loc[y_gas, x_gas] = (
                    all_rank_corr_var_df.at[y_gas, x_gas]
                    + rank_corr_df.at[y_gas, x_gas] ** 2
                )
                # the other parts follow by symmetry
                correlations_df.at[x_gas, y_gas] = correlations_df.at[y_gas, x_gas]
                rank_corr_df.at[x_gas, y_gas] = rank_corr_df.at[y_gas, x_gas]
                all_correlations_df.loc[x_gas, y_gas] = all_correlations_df.at[
                    y_gas, x_gas
                ]
                all_rank_corr_var_df.loc[x_gas, y_gas] = all_rank_corr_var_df.loc[
                    y_gas, x_gas
                ]
                all_rank_corr_df.loc[x_gas, y_gas] = all_rank_corr_df.at[y_gas, x_gas]
            print("Finished x_gas {} in year {}.".format(x_gas, year_of_interest))
        if output_dir is not None:
            correlations_df.to_csv(
                os.path.join(
                    output_dir, "gases_correlation_{}.csv".format(year_of_interest)
                )
            )
            rank_corr_df.to_csv(
                os.path.join(
                    output_dir, "gases_rank_correlation_{}.csv".format(year_of_interest)
                )
            )
    # Complete variance calc by removing mean and dividing through
    all_rank_corr_var_df = (
        all_rank_corr_var_df - len(years) * all_rank_corr_df ** 2
    ) / (len(years) - 1)
    if output_dir is not None:
        all_rank_corr_var_df.to_csv(
            os.path.join(
                output_dir,
                "time_variance_rank_correlation_{}_to_{}.csv".format(
                    min(years), max(years)
                ),
            )
        )
    for gas in df_gases.index:
        all_correlations_df.loc[gas, gas] = np.nan
        all_rank_corr_df.loc[gas, gas] = np.nan
    if output_dir is not None:
        all_correlations_df.to_csv(
            os.path.join(
                output_dir,
                "time_av_absolute_correlation_{}_to_{}.csv".format(
                    min(years), max(years)
                ),
            )
        )
        all_rank_corr_df.to_csv(
            os.path.join(
                output_dir,
                "time_av_absolute_rank_correlation_{}_to_{}.csv".format(
                    min(years), max(years)
                ),
            )
        )
