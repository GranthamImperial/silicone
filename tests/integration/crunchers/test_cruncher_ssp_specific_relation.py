import datetime as dt
import re

import numpy as np
import pandas as pd
import pytest
import scipy.interpolate
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

import silicone.stats
from silicone.database_crunchers import DatabaseCruncherSSPSpecificRelation

_ma = "model_a"
_mb = "model_b"
_mc = "model_c"
_sa = "scen_a"
_sb = "scen_b"
_sc = "scen_c"
_sd = "scen_d"
_se = "scen_e"
_eco2 = "Emissions|CO2"
_gtc = "Gt C/yr"
_ech4 = "Emissions|CH4"
_mtch4 = "Mt CH4/yr"
_ec5f12 = "Emissions|HFC|C5F12"
_ktc5f12 = "kt C5F12/yr"
_ec2f6 = "Emissions|HFC|C2F6"
_ktc2f6 = "kt C2F6/yr"
_msrvu = ["model", "scenario", "region", "variable", "unit"]


class TestDatabaseCruncherSSPSpecificRelation(_DataBaseCruncherTester):
    tclass = DatabaseCruncherSSPSpecificRelation
    # The units in this dataframe are intentionally illogical for C5F12
    tdb = pd.DataFrame(
        [
            [_ma, _sa, "World", _eco2, _gtc, 1, 2, 3, 4],
            [_ma, _sb, "World", _eco2, _gtc, 1, 2, 2, 1],
            [_mb, _sa, "World", _eco2, _gtc, 0.5, 3.5, 3.5, 0.5],
            [_mb, _sb, "World", _eco2, _gtc, 3.5, 0.5, 0.5, 3.5],
            [_ma, _sa, "World", _ech4, _mtch4, 100, 200, 300, 400],
            [_ma, _sb, "World", _ech4, _mtch4, 100, 200, 250, 300],
            [_mb, _sa, "World", _ech4, _mtch4, 220, 260, 250, 230],
            [_mb, _sb, "World", _ech4, _mtch4, 50, 200, 500, 800],
            [_ma, _sa, "World", _ec5f12, _mtch4, 3.14, 4, 5, 6],
            [_ma, _sa, "World", _ec2f6, _ktc2f6, 1.2, 1.5, 1, 0.5],
        ],
        columns=_msrvu + [2010, 2030, 2050, 2070],
    )
    large_db = pd.DataFrame(
        [
            [_ma, _sa, "World", _eco2, _gtc, 1],
            [_ma, _sb, "World", _eco2, _gtc, 5],
            [_mb, _sc, "World", _eco2, _gtc, 0.5],
            [_mb, _sd, "World", _eco2, _gtc, 3.5],
            [_mb, _se, "World", _eco2, _gtc, 0.5],
            [_ma, _sa, "World", _ech4, _mtch4, 100],
            [_ma, _sb, "World", _ech4, _mtch4, 170],
            [_mb, _sc, "World", _ech4, _mtch4, 220],
            [_mb, _sd, "World", _ech4, _mtch4, 50],
            [_mb, _se, "World", _ech4, _mtch4, 150],
        ],
        columns=_msrvu + [2010],
    )

    small_db = pd.DataFrame(
        [[_mb, _sa, "World", _eco2, _gtc, 1.2], [_ma, _sb, "World", _eco2, _gtc, 2.3]],
        columns=_msrvu + [2010],
    )

    tdownscale_df = pd.DataFrame(
        [
            [_mc, _sa, "World", _eco2, _gtc, 1, 2, 3, 4],
            [_mc, _sb, "World", _eco2, _gtc, 0.5, 0.5, 0.5, 0.5],
            [_mc, _sc, "World", _eco2, _gtc, 5, 5, 5, 5],
            [_ma, _sc, "World", _eco2, _gtc, 1.5, 2.5, 2.8, 1.8],
        ],
        columns=_msrvu + [2010, 2030, 2050, 2070],
    )

    simple_df = pd.DataFrame(
        [
            [_mc, _sa, "World", _eco2, _gtc, 0, 1000, 5000],
            [_mc, _sb, "World", _eco2, _gtc, 1, 1000, 5000],
            [_mc, _sa, "World", _ech4, _mtch4, 0, 300, 500],
            [_mc, _sb, "World", _ech4, _mtch4, 1, 300, 500],
        ],
        columns=_msrvu + [2010, 2030, 2050],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|CO2",
            ["Emissions|CH4"],
            required_scenario="scen_a"
        )
        assert callable(res)


    def test_derive_relationship_bad_ssp(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = "There is no data of the appropriate type in the database." \
            " There may be a typo in the SSP option."
        with pytest.raises(ValueError, match=error_msg):
            res = tcruncher.derive_relationship(
                "Emissions|CO2",
                ["Emissions|CH4"],
                required_scenario="Unfindable string"
            )

    def test_derive_relationship_with_nans(self):
        tdb = self.tdb.copy()
        tdb.loc[(tdb["variable"] == _eco2) & (tdb["model"] == _ma), 2050] = np.nan
        tcruncher = self.tclass(IamDataFrame(tdb))
        res = tcruncher.derive_relationship(
            "Emissions|CO2",
            ["Emissions|CH4"],
            required_scenario="scen_a"
        )
        # just make sure that this runs through and no error is raised
        assert callable(res)

    def test_derive_relationship_with_multicolumns(self):
        tdb = self.tdb.copy()
        tcruncher = self.tclass(IamDataFrame(tdb))
        error_msg = re.escape(
            "Having more than one `variable_leaders` is not yet implemented"
        )
        with pytest.raises(NotImplementedError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|CO2", ["Emissions|CH4", "Emissions|HFC|C5F12"], required_scenario="scen_a"
            )

    def test_relationship_usage(self, simple_df):
        tcruncher = self.tclass(simple_df)
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"],required_scenario="scen_a"
        )
        expect_00 = res(simple_df)
        assert expect_00.filter(scenario="scen_a", year=2010)["value"].iloc[0] == 0
        assert expect_00.filter(scenario="scen_b", year=2010)["value"].iloc[0] == 0
        assert all(expect_00.filter(year=2030)["value"] == 1000)
        assert all(expect_00.filter(year=2050)["value"] == 5000)

        # If we include data from scen_b, we then get a slightly different answer
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"], required_scenario=["scen_a", "scen_b"]
        )
        expect_01 = res(simple_df)
        assert expect_01.filter(scenario="scen_a", year=2010)["value"].iloc[0] == 0
        assert expect_01.filter(scenario="scen_b", year=2010)["value"].iloc[0] == 1
        assert all(expect_01.filter(year=2030)["value"] == 1000)
        assert all(expect_01.filter(year=2050)["value"] == 5000)

    def test_numerical_relationship(self):
        # Calculate the values using the cruncher for a fairly detailed dataset
        large_db = IamDataFrame(self.large_db.copy())
        tcruncher = self.tclass(large_db)
        res = tcruncher.derive_relationship(
            "Emissions|CH4",
            ["Emissions|CO2"],
            required_scenario="scen_a"
        )
        assert callable(res)
        to_find = IamDataFrame(self.small_db.copy())
        crunched = res(to_find)

        # Calculate the same values numerically
        xs = large_db.filter(variable="Emissions|CO2")["value"].values
        ys = large_db.filter(variable="Emissions|CH4")["value"].values
        quantile_expected = silicone.stats.rolling_window_find_quantiles(xs, ys, [0.5])
        interpolate_fn = scipy.interpolate.interp1d(
            np.array(quantile_expected.index), quantile_expected.values.squeeze()
        )
        xs_to_interp = to_find.filter(variable="Emissions|CO2")["value"].values

        expected = interpolate_fn(xs_to_interp)

        assert all(crunched["value"].values == expected)

    def test_extreme_values_relationship(self):
        # Our cruncher has a closest-point extrapolation algorithm and therefore
        # should return the same values when filling for data outside tht limits of
        # its cruncher

        # Calculate the values using the cruncher for a fairly detailed dataset
        large_db = IamDataFrame(self.large_db.copy())
        tcruncher = self.tclass(large_db)
        res = tcruncher.derive_relationship("Emissions|CH4", ["Emissions|CO2"])
        assert callable(res)
        crunched = res(large_db)

        # Increase the maximum values
        modify_extreme_db = large_db.filter(variable="Emissions|CO2").copy()
        ind = modify_extreme_db["value"].idxmax
        modify_extreme_db["value"].loc[ind] += 10
        extreme_crunched = res(modify_extreme_db)
        # Check results are the same
        assert all(crunched["value"] == extreme_crunched["value"])
        # Also check that the results are correct
        assert crunched["value"][crunched["scenario"]=="scen_b"].iloc[0] == 170

        # Repeat with reducing the minimum value
        ind = modify_extreme_db["value"].idxmin
        modify_extreme_db["value"].loc[ind] -= 10
        extreme_crunched = res(modify_extreme_db)
        assert all(crunched["value"] == extreme_crunched["value"])
        # There are two smallest points, so we expect to see them equal the mean of the
        # input values for these points
        assert crunched["value"][crunched["scenario"] == "scen_e"].iloc[0] == 185

    def test_derive_relationship_same_gas(self, test_db, test_downscale_df):
        # Given only a single data series, we recreate the original pattern
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"], required_scenario="scen_a")
        crunched = res(test_db)
        assert all(
            abs(
                crunched["value"].reset_index()
                - test_db.filter(variable="Emissions|CO2")["value"].reset_index()
            )
            < 1e15
        )

    def test__make_interpolator(self, test_db):
        variable_leaders = "variable_leaders"
        variable_follower = "variable_follower"
        time_col = "years"
        x_set = np.array([1, 1, 2, 3])
        y_set = np.array([6, 4, 3, 2])
        times = np.array([1, 1, 1, 1])
        wide_db = pd.DataFrame({variable_leaders: x_set, variable_follower: y_set, time_col: times})

        # Illustrate the expected relationship between the numbers above, mapping 1 to
        # the average of 6 and 4, i.e. 5.
        input = np.array([5, 4, 3, 2, 2.5, 1, 0])
        expected_output = np.array([2, 2, 2, 3, 2.5, 5, 5])
        cruncher = self.tclass(test_db)
        interpolator = cruncher._make_interpolator(variable_follower, variable_leaders, wide_db, time_col)
        output = interpolator[1](input)
        assert all(abs(output - expected_output) < 1e-10)

    def test_find_matching_scenarios(self, simple_df):
        variable_leaders = ["Emissions|CO2"]
        variable_follower = "Emissions|CH4"
        time_col = simple_df.time_col
        cruncher = self.tclass(simple_df)
        half_simple_df = simple_df.filter(scenario="scen_a")
        scenarios = cruncher._find_matching_scenarios(half_simple_df, variable_follower, variable_leaders, time_col, ["scen_a", "scen_b"])
        assert scenarios == "scen_a"
        half_simple_df.data["value"].loc[0] = 0.49
        scenarios = cruncher._find_matching_scenarios(half_simple_df, variable_follower,
                                                      variable_leaders, time_col,
                                                      ["scen_a", "scen_b"])
        assert scenarios == "scen_a"
        half_simple_df.data["value"].loc[0] = 0.51
        scenarios = cruncher._find_matching_scenarios(half_simple_df, variable_follower,
                                                      variable_leaders, time_col,
                                                      ["scen_a", "scen_b"])
        assert scenarios == "scen_b"

    def test_find_matching_scenarios_no_data_for_time(self, simple_df):
        variable_leaders = ["Emissions|CO2"]
        variable_follower = "Emissions|CH4"
        time_col = simple_df.time_col
        cruncher = self.tclass(simple_df)
        half_simple_df = simple_df.filter(scenario="scen_a")
        half_simple_df.data[time_col].loc[0] = 0
        with pytest.raises(ValueError):
            cruncher._find_matching_scenarios(half_simple_df, variable_follower,
                                                      variable_leaders, time_col,
                                                      ["scen_a", "scen_b"])

    def test_find_matching_scenarios_complicated(self, test_db, simple_df):
        #TODO: check that this is true
        variable_leaders = ["Emissions|CO2"]
        variable_follower = "Emissions|CH4"
        test_db = self._adjust_time_style_to_match(test_db, simple_df)
        time_col = simple_df.time_col
        cruncher = self.tclass(test_db)
        scenarios = cruncher._find_matching_scenarios(simple_df, variable_follower,
                                                      variable_leaders, time_col,
                                                      ["scen_a", "scen_b"])
        assert scenarios == "scen_b"

    def test_derive_relationship_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|CO2"]
        tcruncher = self.tclass(test_db.filter(variable=variable_leaders, keep=False))

        error_msg = re.escape(
            "No data for `variable_leaders` ({}) in database".format(variable_leaders)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|CH4", variable_leaders, required_scenario="scen_a")

    def test_crunch_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|CO2"]
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CH4", variable_leaders, required_scenario="scen_a")
        error_msg = re.escape(
            "There is no data for {} so it cannot be infilled".format(
                variable_leaders
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            res(test_db.filter(variable=variable_leaders, keep=False))

    def test_derive_relationship_error_no_info_follower(self, test_db):
        # test that crunching fails if there's no data about the follower gas in the
        # database
        variable_follower = "Emissions|CH4"
        tcruncher = self.tclass(test_db.filter(variable=variable_follower, keep=False))

        error_msg = re.escape(
            "No data for `variable_follower` ({}) in database".format(variable_follower)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable_follower, ["Emissions|CO2"], required_scenario="scen_a")

    def test_relationship_usage_wrong_unit(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"], required_scenario="scen_a")

        exp_units = test_db.filter(variable="Emissions|CO2")["unit"].iloc[0]

        wrong_unit = "t C/yr"
        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        test_downscale_df["unit"] = wrong_unit

        error_msg = re.escape(
            "Units of lead variable is meant to be `{}`, found `{}`".format(
                exp_units, wrong_unit
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            res(test_downscale_df)

    def test_relationship_usage_wrong_time(self):
        tdb = IamDataFrame(self.tdb)
        tcruncher = self.tclass(tdb)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"], required_scenario="scen_a")

        test_downscale_df = IamDataFrame(self.tdb).timeseries()
        test_downscale_df.columns = test_downscale_df.columns.map(
            lambda x: dt.datetime(x, 1, 1)
        )
        test_downscale_df = IamDataFrame(test_downscale_df)

        error_msg = re.escape(
            "`in_iamdf` time column must be the same as the time column used "
            "to generate this filler function (`year`)"
        )
        with pytest.raises(ValueError, match=error_msg):
            res(test_downscale_df)

    def test_relationship_usage_insufficient_timepoints(
        self, test_db, test_downscale_df
    ):
        tcruncher = self.tclass(test_db.filter(year=2030, keep=False))

        filler = tcruncher.derive_relationship("Emissions|CH4", ["Emissions|CO2"], required_scenario="scen_a")

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)

        error_msg = re.escape(
            "Not all required timepoints are present in the database we "
            "crunched, we crunched \n\t`{}`\nbut you passed in \n\t{}".format(
                list(
                    test_db.filter(year=2030, keep=False).timeseries().columns.tolist()
                ),
                test_db.timeseries().columns.tolist(),
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)
