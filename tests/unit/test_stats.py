import os
import re

import numpy as np
import pandas as pd
import pyam
import pytest
import scipy.interpolate

import silicone.stats as stats

_mc = "model_c"
_sa = "scen_a"
_sb = "scen_b"
_sc = "scen_c"
_eco2 = "Emissions|CO2"
_gtc = "Gt C/yr"
_ech4 = "Emissions|CH4"
_mtch4 = "Mt CH4/yr"
_msrvu = ["model", "scenario", "region", "variable", "unit"]
simple_df = pd.DataFrame(
    [
        [_mc, _sa, "World", _eco2, _gtc, 0, 200, 1],
        [_mc, _sb, "World", _eco2, _gtc, 2, 100, -1],
        [_mc, _sa, "World", _ech4, _mtch4, 0, 300, 1],
        [_mc, _sb, "World", _ech4, _mtch4, 2, 600, -1],
        [_mc, _sc, "World", _eco2, _gtc, np.nan, np.nan, 0.5],
        [_mc, _sc, "World", _ech4, _mtch4, np.nan, np.nan, 0.5],
    ],
    columns=_msrvu + [2010, 2030, 2050],
)
simple_df = pyam.IamDataFrame(simple_df)


@pytest.mark.parametrize(
    "xs,ys",
    (
        (np.array([0, 0, 1, 1]), np.array([0, 1, 0, 1])),
        (np.array([0, 0, 1, 1]), np.array([0, 1, 1, 0])),
        (np.array([0, 1, 0, 1]), np.array([0, 1, 1, 0])),
    ),
)
def test_rolling_window_find_quantiles(xs, ys):
    desired_quantiles = [0.4, 0.5, 0.6]
    # Firstly take the window centre at a lead value of 0. With a
    # decay_length_factor=20 and nwindows=10, the points at a lead value
    # of 0 are 10 window centres away hence receive a weight of 1/2 relative
    # to the points at a lead value of 0.
    # with the points in order of follow values then ordered by lead
    # values where lead values are the same we have i.e. the points are:
    # points: [(0, 0), (1, 0), (0, 1), (1, 1)]
    # we have
    # unnormalised weights: [2, 1, 2, 1]
    # normalised weights are: [1/3, 1/6, 1/3, 1/6]
    # cumulative weights are hence: [2/6, 3/6, 5/6, 1]
    # subtracting half the weights we get: [1/6, 5/12, 4/6, 11/12]
    # Hence above quantiles of  quantiles of 5/12, we have a gradient (in
    # follower value - quantile space) = (1 - 0) / (4/6 - 5/12)
    # thus our relationship is (quant - 5/12) * grad
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 11, 20)
    assert np.allclose(
        quantiles.iloc[0].tolist(),
        np.array([0, (0.5 - 5 / 12), (0.6 - 5 / 12)]) * 1 / (4 / 6 - 5 / 12),
    )
    # At the far side, we have switched the weights around, so that cumulative weights
    # are 1/12 and 1/3 for y = 0 and 7 / 12 and 5 / 12 for y = 1.
    assert np.allclose(
        quantiles.iloc[-1].tolist(), [(0.4 - 1 / 3) * 4, (0.5 - 1 / 3) * 4, 1]
    )

    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 0, 1, 1])
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 11, 20)
    # And x = 0, a gradient of 4 starting from 1/2 at q > 0.5
    assert np.allclose(
        quantiles.iloc[0].tolist(),
        [0, 0, 0.1 * 4],
    )
    # at x = 1 we have the exact opposite
    assert np.allclose(
        quantiles.iloc[-1, :].tolist(),
        [(0.4 - 1 / 4) * 4, 1, 1],
    )

    desired_quantiles = [0, 0.5, 1]
    quantiles = stats.rolling_window_find_quantiles(
        np.array([1]), np.array([1]), desired_quantiles, 11, 20
    )
    assert all(quantiles.iloc[0, :] == [1, 1, 1])

    desired_quantiles = [0, 0.5, 1]
    quantiles = stats.rolling_window_find_quantiles(
        np.array([1, 1]), np.array([1, 1]), desired_quantiles, 11, 20
    )
    assert all(quantiles.iloc[0, :] == [1, 1, 1])


def test_rolling_window_find_quantiles_same_points():
    # If all the x-values are the same, this should just be our interpretation of
    # quantiles at all points
    xs = np.array([1] * 11)
    ys = np.array(range(11))
    desired_quantiles = [0, 0.4, 0.5, 0.6, 0.85, 1]
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 11, 20)

    cumsum_weights = (0.5 + np.arange(11)) / 11
    calculated_quantiles = []
    for quant in desired_quantiles:
        calculated_quantiles.append(
            scipy.interpolate.interp1d(
                cumsum_weights,
                ys,
                bounds_error=False,
                fill_value=(ys[0], ys[-1]),
                assume_sorted=True,
            )(quant)
        )

    assert np.allclose(quantiles.squeeze().tolist(), calculated_quantiles)


def test_rolling_window_find_quantiles_one():
    # If all the x-values are the same, this should just be our interpretation of
    # quantiles at all points
    xs = np.array([1])
    ys = np.array([2])
    desired_quantiles = [0, 0.4, 0.5, 0.6, 0.85, 1]
    quantiles = stats.rolling_window_find_quantiles(
        xs, ys, desired_quantiles, 11, 2 * 9
    )

    assert np.allclose(quantiles.values.squeeze(), 2)


def test_calc_quantiles_of_data():
    # We want to include the value 100 at the end of this range, so go to 101.
    range100 = pd.Series(np.arange(0, 101))
    to_quant = pd.Series([-1, 0, 10, 50, 99, 100, 101])
    expected = np.array([0, 0, 0.1, 0.5, 0.99, 1, 1])
    res = stats.calc_quantiles_of_data(range100, to_quant)
    assert np.allclose(res, expected)
    # Ensure that we're not using the index in the calculation
    range100_rev = pd.Series(np.arange(100, -1, -1))
    res = stats.calc_quantiles_of_data(range100_rev, to_quant)
    assert np.allclose(res, expected)
    # Test that uniform weighting does not affect the values
    weights = pd.Series([30 for i in range(len(range100))], index=range100_rev.index)
    res = stats.calc_quantiles_of_data(range100_rev, to_quant, weighting=weights)
    assert np.allclose(res, expected)


def test_calc_quantiles_weighted():
    # Test that using weighted quantiles produces the expected results
    between1and10 = pd.Series([0, 1, 10], index=["nought", "one", "ten"])
    small = 0.00001
    weights = pd.Series([small, 1, 1], index=["nought", "one", "ten"])
    to_quant = pd.Series([0, 1, 1.1, 5, 7, 10, 11])
    # We expect: 1 should be small/ (1+small) exactly. higher values are this plus the
    # remaining distance interpolated between 1 and 10.
    smaller = small / (1 + small)
    expected = np.array(
        [0, smaller, smaller + 0.1 / 9, smaller + 4 / 9, smaller + 6 / 9, 1, 1]
    )
    res = stats.calc_quantiles_of_data(between1and10, to_quant, weighting=weights)
    assert np.allclose(res, expected)

    # The weights value can be entered in any order
    weights = pd.Series([1, 1, small], index=["one", "ten", "nought"])
    res = stats.calc_quantiles_of_data(between1and10, to_quant, weighting=weights)
    assert np.allclose(res, expected)


def test_calc_quantiles_weighted_nans():
    # Test that adding nan values to the system doesn't affect results
    between1and10 = pd.Series([0, 1, np.nan, 10], index=["nought", "one", "non", "ten"])
    small = 0.00001
    weights = pd.Series([small, 1, 1, 10], index=["nought", "one", "ten", "non"])
    to_quant = pd.Series([0, 1, 1.1, 5, 7, 10, 11])
    # We expect: 1 should be small/ (1+small) exactly. higher values are this plus the
    # remaining distance interpolated between 1 and 10.
    smaller = small / (1 + small)
    expected = np.array(
        [0, smaller, smaller + 0.1 / 9, smaller + 4 / 9, smaller + 6 / 9, 1, 1]
    )
    res = stats.calc_quantiles_of_data(between1and10, to_quant, weighting=weights)
    assert np.allclose(res, expected)


def test_calc_quantile_bad_weights():
    between1and10 = pd.Series([0, 1, np.nan, 10], index=["nought", "one", "non", "ten"])
    weights = [0.1, 1, 1, 10, 1]
    to_quant = pd.Series([0, 1, 1.1, 5, 7, 10, 11])
    msg = "The weighting variable should be a Series"
    with pytest.raises(AssertionError, match=msg):
        stats.calc_quantiles_of_data(between1and10, to_quant, weighting=weights)
    weights = pd.Series(weights, index=["nought", "one", "ten", "non", "o"])
    msg = (
        "There must be the same number of weights as entries in the database. "
        "We have len {} in weights and len {} in distribution.".format(
            len(weights), len(between1and10)
        )
    )
    with pytest.raises(AssertionError, match=msg):
        stats.calc_quantiles_of_data(between1and10, to_quant, weighting=weights)
    weights = weights.iloc[1:]
    msg = re.escape(
        "The entries in the weighting Series are not clearly aligned with the "
        "distribution Series. We have \n {} \n in the weighting and \n {} \n in "
        "the distribution ".format(weights.index, between1and10.index)
    )
    with pytest.raises(AssertionError, match=msg):
        stats.calc_quantiles_of_data(between1and10, to_quant, weighting=weights)


def test_calc_quantiles_of_data_smooth():
    # We want to include the value 100 at the end of this range, so go to 101.
    range100 = pd.Series(np.arange(0, 101))
    to_quant = pd.Series([-1, 0, 10, 50, 99, 100, 101])
    expected = np.array([0, 0, 0.1, 0.5, 0.99, 1, 1])
    # In the limit of no smoothness we should recover the original result
    res = stats.calc_quantiles_of_data(range100, to_quant, 0.001)
    assert np.allclose(res, expected, atol=0.01, rtol=0.01)

    # But when we add smoothness, we encounter slight, symmetric differences
    res_smooth = stats.calc_quantiles_of_data(range100, to_quant, 0.3)
    assert np.allclose(res[0] + res[-1], res_smooth[0] + res_smooth[-1], atol=5e-4)
    assert np.allclose(res[1] + res[-2], res_smooth[1] + res_smooth[-2], atol=5e-3)
    assert all([res_smooth[i] >= res[i] for i in range(3)])
    assert all([res_smooth[i] <= res[i] for i in range(4, 7)])
    # Substantially the same results should be found when using the "scott" smoothing
    res_smooth = stats.calc_quantiles_of_data(range100, to_quant, "scott")
    assert np.allclose(res_smooth[3], 0.5, atol=0.003)
    assert np.allclose(res[0] + res[-1], res_smooth[0] + res_smooth[-1], atol=5e-4)
    assert all([res_smooth[i] >= res[i] for i in range(3)])
    assert all([res_smooth[i] <= res[i] for i in range(4, 7)])
    assert all([res_smooth[i] >= res[i] for i in range(3)])


@pytest.mark.parametrize("smoothing", (None, 0.3))
def test_calc_quantiles_of_insufficient_data(smoothing):
    message = "No valid data entered to establish the quantiles."
    nans = pd.Series(np.nan)
    single_val = pd.Series(1)
    to_quant = pd.Series([-1, 0, 10, 50, 99, 100, 101])
    with pytest.raises(ValueError, match=message):
        stats.calc_quantiles_of_data(nans, to_quant, smoothing)
    res = stats.calc_quantiles_of_data(single_val, to_quant, smoothing)
    assert all(np.isnan(res))
    assert len(res) == len(to_quant)


def test_calc_all_emissions_correlations_works(tmpdir):
    # We test that this saves a file in the correct place, with the correct results
    test_folder = os.path.join(tmpdir, "output")
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    stats.calc_all_emissions_correlations(
        simple_df, list(set(simple_df["year"])), test_folder
    )
    expected = {2010: 1, 2030: -1, 2050: 1}
    for year in list(set(simple_df["year"])):
        for file_string in ["gases_correlation", "gases_rank_correlation"]:
            test_file = os.path.join(test_folder, file_string + "_{}.csv".format(year))
            assert os.path.isfile(test_file)
            test_results = pd.read_csv(test_file)
            assert np.isnan(test_results.iloc[0].iloc[1])
            assert test_results.iloc[1].iloc[1] == expected.get(year)
            assert test_results.iloc[0].iloc[2] == expected.get(year)
            os.remove(test_file)
            assert not os.path.isfile(test_file)
    for file_string in [
        "time_av_absolute_correlation",
        "time_av_absolute_rank_correlation",
        "time_variance_rank_correlation",
    ]:
        test_file = os.path.join(
            test_folder,
            file_string
            + "_{}_to_{}.csv".format(
                min(set(simple_df["year"])), max(set(simple_df["year"]))
            ),
        )
        assert os.path.isfile(test_file)
        test_results = pd.read_csv(test_file)
        if file_string == "time_variance_rank_correlation":
            # All values are zeros since the abs value is 1 in all cases (+/-1)
            assert np.allclose(test_results.iloc[0].iloc[1], 0)
            assert np.allclose(test_results.iloc[1].iloc[1], 0)
            assert np.allclose(test_results.iloc[0].iloc[2], 0)
        else:
            assert np.isnan(test_results.iloc[0].iloc[1])
            assert np.allclose(test_results.iloc[1].iloc[1], 1)
            assert np.allclose(test_results.iloc[0].iloc[2], 1)
        os.remove(test_file)
        assert not os.path.isfile(test_file)
    # Check that the variable counts are correct too.
    test_file = os.path.join(test_folder, "variable_counts.csv")
    assert os.path.isfile(test_file)
    test_results = pd.read_csv(test_file)
    assert np.allclose(test_results["0"].iloc[0], 3)
    assert np.allclose(test_results["0"].iloc[1], 3)
    os.remove(test_file)
    assert not os.path.isfile(test_file)


def test_calc_all_emissions_numerical(tmpdir):
    # We construct a specific situation and check that the numerical answers are correct
    test_folder = os.path.join(tmpdir, "output")
    if not os.path.isdir(test_folder):
        os.makedirs(test_folder)
    # We establish a more complicated set of values
    numerical_df = simple_df.copy().data
    numerical_df["model"] = numerical_df["model"] + numerical_df["year"].map(
        lambda x: str(x)
    )
    numerical_df["year"] = 2010
    numerical_df = pyam.IamDataFrame(numerical_df)
    # Perform the calculations
    stats.calc_all_emissions_correlations(numerical_df, [2010], test_folder)
    # The order of the elements is identical for the different cases, no sorting needed
    xs = numerical_df.filter(variable=_eco2).data["value"].values
    ys = numerical_df.filter(variable=_ech4).data["value"].values

    def calc_correl(x, y):
        xmean = sum(x) / len(x)
        ymean = sum(y) / len(y)
        return (
            sum((x - xmean) * (y - ymean))
            / (sum((x - xmean) ** 2) * sum((y - ymean) ** 2)) ** 0.5
        )

    correl = calc_correl(xs, ys)
    test_file = os.path.join(test_folder, "gases_correlation" + "_{}.csv".format(2010))
    test_results = pd.read_csv(test_file)
    assert np.isclose(test_results.iloc[1].iloc[1], correl)
    os.remove(test_file)
    x_ord = np.argsort(xs)
    y_ord = np.argsort(ys)
    rank_correl = calc_correl(x_ord, y_ord)
    test_file = os.path.join(
        test_folder, "gases_rank_correlation" + "_{}.csv".format(2010)
    )
    test_results = pd.read_csv(test_file)
    assert np.isclose(test_results.iloc[1].iloc[1], rank_correl, rtol=1e-4)
    os.remove(test_file)
    for file_string in [
        "time_av_absolute_correlation",
        "time_av_absolute_rank_correlation",
        "time_variance_rank_correlation",
    ]:
        test_file = os.path.join(
            test_folder,
            file_string
            + "_{}_to_{}.csv".format(
                min(set(numerical_df["year"])), max(set(numerical_df["year"]))
            ),
        )
        test_results = pd.read_csv(test_file)
        some_cor = rank_correl if file_string.__contains__("rank") else correl
        if file_string == "time_variance_rank_correlation":
            assert np.isnan(test_results.iloc[1].iloc[1])
        else:
            assert np.isclose(test_results.iloc[1].iloc[1], some_cor, rtol=1e-4)
        os.remove(test_file)
    test_file = os.path.join(test_folder, "variable_counts.csv")
    assert os.path.isfile(test_file)
    test_results = pd.read_csv(test_file)
    assert np.allclose(test_results["0"].iloc[0], 7)
    assert np.allclose(test_results["0"].iloc[1], 7)
    os.remove(test_file)
    assert not os.path.isfile(test_file)
    # Now do a test for just the variance. This requires multiple years
    numerical_df = numerical_df.data
    numerical_df["value"] += 10
    numerical_df = pd.concat((numerical_df, simple_df.data))
    numerical_df["year"] = numerical_df["year"].map(lambda x: int(x))
    numerical_df = pyam.IamDataFrame(numerical_df)
    rank_cors = []
    years = [2010, 2030, 2050]
    for year in years:
        xs = numerical_df.filter(variable=_eco2, year=year).data["value"].values
        ys = numerical_df.filter(variable=_ech4, year=year).data["value"].values
        x_ord = np.argsort(xs)
        y_ord = np.argsort(ys)
        rank_cors.append(abs(calc_correl(x_ord, y_ord)))
    expect_var = np.var(rank_cors, ddof=1)
    stats.calc_all_emissions_correlations(numerical_df, years, test_folder)
    for file_string in [
        "time_av_absolute_correlation",
        "time_av_absolute_rank_correlation",
        "time_variance_rank_correlation",
    ]:
        test_file = os.path.join(
            test_folder,
            file_string
            + "_{}_to_{}.csv".format(
                min(set(simple_df["year"])), max(set(simple_df["year"]))
            ),
        )
        test_results = pd.read_csv(test_file)
        if file_string == "time_variance_rank_correlation":
            assert np.isclose(expect_var, test_results.iloc[1].iloc[1])
        os.remove(test_file)


def test_quantile_values_not_above_1():
    # We construct a scenario that can return quantile values above 1 if the numerical
    # sums are not handled properly
    to_quant = pd.Series(
        [38751.41118, 40656.52654, 40656.52654, 40656.52654, 40656.52654, 42656.04102]
    )
    fringe = (max(to_quant) - min(to_quant)) * 2
    infiller = pd.Series(
        np.arange(min(to_quant) - fringe, max(to_quant) + fringe, 1000)
    )
    res = stats.calc_quantiles_of_data(to_quant, infiller, 0.3)
    assert max(res) == 1
    assert min(res) == 0
