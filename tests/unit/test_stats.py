import numpy as np
import pyam
import pandas as pd

import silicone.stats as stats
import os

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

def test_rolling_window_find_quantiles():
    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 1, 0, 1])
    desired_quantiles = [0.4, 0.5, 0.6]
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2 * 9)
    assert all(quantiles.iloc[0] == [0, 0, 1])
    assert all(quantiles.iloc[1] == [0, 0, 1])

    xs = np.array([0, 0, 1, 1])
    ys = np.array([0, 0, 1, 1])
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2 * 9)
    assert all(quantiles.iloc[0, :] == 0)
    assert all(quantiles.iloc[-1, :] == 1)
    assert all(quantiles.iloc[5, :] == [0, 0, 1])

    desired_quantiles = [0, 0.5, 1]
    quantiles = stats.rolling_window_find_quantiles(
        np.array([1]), np.array([1]), desired_quantiles, 9, 2 * 9
    )
    assert all(quantiles.iloc[0, :] == [1, 1, 1])

    desired_quantiles = [0, 0.5, 1]
    quantiles = stats.rolling_window_find_quantiles(
        np.array([1, 1]), np.array([1, 1]), desired_quantiles, 9, 2 * 9
    )
    assert all(quantiles.iloc[0, :] == [1, 1, 1])


def test_rolling_window_find_quantiles_same_points():
    # If all the x-values are the same, this should just be our interpretation of
    # quantiles at all points
    xs = np.array([1] * 11)
    ys = np.array(range(11))
    desired_quantiles = [0, 0.4, 0.5, 0.6, 0.85, 1]
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2 * 9)

    cumsum_weights = (1 + np.arange(11)) / 11
    calculated_quantiles = []
    for quant in desired_quantiles:
        calculated_quantiles.append(min(ys[cumsum_weights >= quant]))

    assert all(quantiles.values.squeeze() == calculated_quantiles)


def test_rolling_window_find_quantiles_one():
    # If all the x-values are the same, this should just be our interpretation of
    # quantiles at all points
    xs = np.array([1])
    ys = np.array([2])
    desired_quantiles = [0, 0.4, 0.5, 0.6, 0.85, 1]
    quantiles = stats.rolling_window_find_quantiles(xs, ys, desired_quantiles, 9, 2 * 9)

    assert np.allclose(quantiles.values.squeeze(), 2)


def test_calc_all_emissions_correlations_works():
    # We test that this saves a file in the correct place, with the correct results
    test_folder = "./"
    stats.calc_all_emissions_correlations(
        simple_df, list(set(simple_df["year"])), test_folder
    )
    expected = {2010: 1, 2030: -1, 2050: 1}
    for year in list(set(simple_df["year"])):
        for file_string in ["gases_correlation", "gases_rank_correlation"]:
            test_file = test_folder + file_string + "_{}.csv".format(year)
            assert os.path.isfile(test_file)
            test_results = pd.read_csv(test_file)
            assert np.isnan(test_results.iloc[0].iloc[1])
            assert test_results.iloc[1].iloc[1] == expected.get(year)
            assert test_results.iloc[0].iloc[2] == expected.get(year)
            os.remove(test_file)
            assert not os.path.isfile(test_file)
    for file_string in ["time_av_gases_correlation", "time_av_gases_rank_correlation"]:
        test_file = test_folder + file_string + "_{}_to_{}.csv".format(
            min(set(simple_df["year"])), max(set(simple_df["year"]))
        )
        assert os.path.isfile(test_file)
        test_results = pd.read_csv(test_file)
        assert np.isnan(test_results.iloc[0].iloc[1])
        assert np.allclose(test_results.iloc[1].iloc[1], 1/3)
        assert np.allclose(test_results.iloc[0].iloc[2], 1 / 3)
        os.remove(test_file)
        assert not os.path.isfile(test_file)

def test_calc_all_emissions_numerical():
    # We construct a specific situation and check that the numerical answers are correct
    test_folder = "./"
    # We establish a more complicated set of values
    numerical_df = simple_df
    numerical_df.data["model"] = numerical_df.data["model"] + numerical_df.data["year"].map(lambda x: str(x))
    numerical_df.data["year"] = 2010
    numerical_df = pyam.IamDataFrame(numerical_df.data)
    # Perform the calculations
    stats.calc_all_emissions_correlations(
        numerical_df, [2010], test_folder
    )
    # The order of the elements is identical for the different cases, no sorting needed
    xs = numerical_df.filter(variable=_eco2).data["value"].values
    ys = numerical_df.filter(variable=_ech4).data["value"].values

    def calc_correl(x, y):
        xmean = sum(x) / len(x)
        ymean = sum(y) / len(y)
        return sum((x - xmean) * (y - ymean)) / (
            sum((x - xmean) ** 2) * sum((y - ymean) ** 2)
        ) ** 0.5

    correl = calc_correl(xs, ys)
    test_file = test_folder + "gases_correlation" + "_{}.csv".format(2010)
    test_results = pd.read_csv(test_file)
    assert np.isclose(test_results.iloc[1].iloc[1], correl)
    # our ordering starts from 1
    x_ord = np.argsort(xs)
    y_ord = np.argsort(ys)
    rank_correl = calc_correl(x_ord, y_ord)
    test_file = test_folder + "gases_rank_correlation" + "_{}.csv".format(2010)
    test_results = pd.read_csv(test_file)
    assert np.isclose(test_results.iloc[1].iloc[1], rank_correl, rtol=1e-4)
