import datetime as dt
import re

import numpy as np
import pandas as pd
import pytest
import scipy.interpolate
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import (
    Interpolation,
    LinearInterpolation,
    ScenarioAndModelSpecificInterpolate,
)

_ma = "model_a"
_mb = "model_b"
_mc = "model_c"
_sa = "scen_a"
_sb = "scen_b"
_sc = "scen_c"
_sd = "scen_d"
_se = "scen_e"
_sf = "scen_e"
_eco2 = "Emissions|CO2"
_gtc = "Gt C/yr"
_ech4 = "Emissions|CH4"
_mtch4 = "Mt CH4/yr"
_ec5f12 = "Emissions|HFC|C5F12"
_ktc5f12 = "kt C5F12/yr"
_ec2f6 = "Emissions|HFC|C2F6"
_ktc2f6 = "kt C2F6/yr"
_msrvu = ["model", "scenario", "region", "variable", "unit"]


class TestDatabaseCruncherScenarioAndModelSpecificInterpolate(_DataBaseCruncherTester):
    tclass = ScenarioAndModelSpecificInterpolate
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

    def test_database_crunchers_with_filters(self, test_db, simple_df):
        test_db = self._adjust_time_style_to_match(test_db, simple_df)
        tcruncher_filtered = self.tclass(test_db)
        tcruncher_generic = Interpolation(test_db)
        tcruncher_linear = LinearInterpolation(test_db)
        filtered_cruncher = tcruncher_filtered.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"]
        )
        infilled_filter = filtered_cruncher(simple_df)
        linear_cruncher = tcruncher_linear.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"]
        )
        infilled_linear = linear_cruncher(simple_df)
        generic_cruncher = tcruncher_generic.derive_relationship("Emissions|CO2", ["Emissions|CH4"], interpkind="linear")
        infilled_generic = generic_cruncher(simple_df)
        pd.testing.assert_frame_equal(infilled_filter.data, infilled_linear.data)
        pd.testing.assert_frame_equal(infilled_generic.data, infilled_linear.data)

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"], required_scenario="scen_a"
        )
        assert callable(res)

    def test_derive_relationship_bad_ssp(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = (
            "There is no data of the appropriate type in the database."
            " There may be a typo in the SSP option."
        )
        with pytest.raises(ValueError, match=error_msg):
            res = tcruncher.derive_relationship(
                "Emissions|CO2",
                ["Emissions|CH4"],
                required_scenario="Unfindable string",
            )

    def test_derive_relationship_with_nans(self):
        tdb = self.tdb.copy()
        tdb.loc[(tdb["variable"] == _eco2) & (tdb["model"] == _ma), 2050] = np.nan
        tcruncher = self.tclass(IamDataFrame(tdb))
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"], required_scenario="scen_a"
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
                "Emissions|CO2",
                ["Emissions|CH4", "Emissions|HFC|C5F12"],
                required_scenario="scen_a",
            )

    @pytest.mark.parametrize("add_col", [None, "extra_col"])
    def test_relationship_usage(self, simple_df, add_col):
        tcruncher = self.tclass(simple_df)
        lead = ["Emissions|CH4"]
        follow = "Emissions|CO2"
        res = tcruncher.derive_relationship(follow, lead, required_scenario="scen_a")
        if add_col:
            add_col_val = "blah"
            simple_df = simple_df.data
            simple_df[add_col] = add_col_val
            simple_df = IamDataFrame(simple_df)
            assert simple_df.extra_cols[0] == add_col

        expect_00 = res(simple_df)
        assert expect_00.filter(scenario="scen_a", year=2010).data["value"].iloc[0] == 0
        assert expect_00.filter(scenario="scen_b", year=2010).data["value"].iloc[0] == 0
        assert all(expect_00.filter(year=2030).data["value"] == 1000)
        assert all(expect_00.filter(year=2050).data["value"] == 5000)

        # If we include data from scen_b, we then get a slightly different answer
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"], required_scenario=["scen_a", "scen_b"]
        )
        expect_01 = res(simple_df)
        assert expect_01.filter(scenario="scen_a", year=2010).data["value"].iloc[0] == 0
        assert expect_01.filter(scenario="scen_b", year=2010).data["value"].iloc[0] == 1
        assert all(expect_01.filter(year=2030).data["value"] == 1000)
        assert all(expect_01.filter(year=2050).data["value"] == 5000)

        # Test we can append our answer
        append_df = simple_df.filter(variable=lead).append(expect_01)
        assert append_df.filter(variable=follow).equals(expect_01)

        if add_col:
            assert all(append_df[add_col] == add_col_val)


    @pytest.mark.parametrize("interpkind", [
        "linear", "quadratic", "nearest", "PchipInterpolator"
    ])
    def test_working_with_repeat_values(self, test_db, interpkind):
        # Do the crunchers work when all values are the same in both the input and
        # output?
        rep_db = test_db.copy().data
        rep_db["value"] = 10
        rep_db_2 = rep_db.copy()
        rep_db_2["value"] = 2
        rep_db = IamDataFrame(rep_db)
        rep_db_2 = IamDataFrame(rep_db_2)
        tcruncher = self.tclass(IamDataFrame(rep_db))
        lead = ["Emissions|CH4"]
        follow = "Emissions|CO2"
        res = tcruncher.derive_relationship(follow, lead, interpkind=interpkind)(
            rep_db_2
        )
        assert np.allclose(res.data.value, 10)



    @pytest.mark.parametrize("interpkind", [
        "linear", "quadratic", "nearest", "PchipInterpolator"
    ])
    def test_numerical_relationship(self, interpkind):
        # Calculate the values using the cruncher for a fairly detailed dataset
        large_db = IamDataFrame(self.large_db.copy())
        tcruncher = self.tclass(large_db)
        res = tcruncher.derive_relationship(
            "Emissions|CH4",
            ["Emissions|CO2"],
            required_scenario="*",
            interpkind=interpkind,
        )
        assert callable(res)
        to_find = IamDataFrame(self.small_db.copy())
        crunched = res(to_find)

        # Calculate the same values numerically
        xs = large_db.filter(variable="Emissions|CO2").data["value"].values
        ys = large_db.filter(variable="Emissions|CH4").data["value"].values
        ys = [np.mean(ys[xs == x]) for x in xs]
        # We must remove the duplicate value from the lists
        if interpkind != "PchipInterpolator":
            interpolate_fn = scipy.interpolate.interp1d(xs[:-1], ys[:-1], kind=interpkind)
        else:
            xsort = np.argsort(xs[:-1])
            interpolate_fn = scipy.interpolate.PchipInterpolator(xs[xsort], np.array(ys)[xsort])
        xs_to_interp = to_find.filter(variable="Emissions|CO2").data["value"].values

        expected = interpolate_fn(xs_to_interp)

        assert all(crunched.data["value"].values == expected)

    @pytest.mark.parametrize("interpkind", [
        "linear", "quadratic", "nearest", "PchipInterpolator"
    ])
    def test_extreme_values_relationship(self, interpkind):
        # Our cruncher has a closest-point extrapolation algorithm and therefore
        # should return the same values when filling for data outside tht limits of
        # its cruncher

        # Calculate the values using the cruncher for a fairly detailed dataset
        large_db = IamDataFrame(self.large_db.copy())
        tcruncher = self.tclass(large_db)
        res = tcruncher.derive_relationship("Emissions|CH4", ["Emissions|CO2"], interpkind=interpkind)
        assert callable(res)
        crunched = res(large_db)

        # Increase the maximum values
        modify_extreme_db = large_db.filter(variable="Emissions|CO2").copy().data
        ind = modify_extreme_db["value"].idxmax()
        modify_extreme_db.loc[ind, "value"] += 10
        modify_extreme_db = IamDataFrame(modify_extreme_db)
        extreme_crunched = res(modify_extreme_db)
        # Check results are the same
        assert all(crunched.data["value"] == extreme_crunched.data["value"])
        # Also check that the results are correct
        assert crunched.data["value"][crunched["scenario"] == "scen_b"].iloc[0] == 170

        # Repeat with reducing the minimum value
        modify_extreme_db = modify_extreme_db.data
        ind = modify_extreme_db["value"].idxmin()
        modify_extreme_db.loc[ind, "value"] -= 10
        modify_extreme_db = IamDataFrame(modify_extreme_db)
        extreme_crunched = res(modify_extreme_db)
        assert all(crunched.data["value"] == extreme_crunched.data["value"])
        # There are two smallest points, so we expect to see them equal the mean of the
        # input values for these points
        assert crunched.data["value"][crunched["scenario"] == "scen_e"].iloc[0] == 185

    def test_derive_relationship_same_gas(self, test_db, test_downscale_df):
        # Given only a single data series, we recreate the original pattern
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CO2"], required_scenario="scen_a"
        )
        crunched = res(test_db)
        assert all(
            abs(
                crunched.data["value"].reset_index()
                - test_db.filter(variable="Emissions|CO2").data["value"].reset_index()
            )
            < 1e15
        )

    def test_derive_relationship_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|CO2"]
        tcruncher = self.tclass(test_db.filter(variable=variable_leaders, keep=False))

        error_msg = re.escape(
            "No data for `variable_leaders` ({}) in database".format(variable_leaders)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|CH4", variable_leaders, required_scenario="scen_a"
            )

    def test_crunch_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|CO2"]
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|CH4", variable_leaders, required_scenario="scen_a"
        )
        error_msg = re.escape(
            "There is no data for {} so it cannot be infilled".format(variable_leaders)
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
            tcruncher.derive_relationship(
                variable_follower, ["Emissions|CO2"], required_scenario="scen_a"
            )

    def test_relationship_no_data(self, test_db):
        tcruncher = self.tclass(test_db)
        err_mssg = "There is no data of the appropriate type in the database."
        with pytest.raises(ValueError, match=err_mssg):
            tcruncher.derive_relationship(
                "Silly name", ["other silly name"], required_scenario="scen_a"
            )

    def test_relationship_usage_wrong_unit(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CO2"], required_scenario="scen_a"
        )

        exp_units = test_db.filter(variable="Emissions|CO2")["unit"].iloc[0]

        wrong_unit = "t C/yr"
        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df, test_db
        ).data
        test_downscale_df["unit"] = wrong_unit
        test_downscale_df = IamDataFrame(test_downscale_df)

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
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CO2"], required_scenario="scen_a"
        )

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

        filler = tcruncher.derive_relationship(
            "Emissions|CH4", ["Emissions|CO2"], required_scenario="scen_a"
        )

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
