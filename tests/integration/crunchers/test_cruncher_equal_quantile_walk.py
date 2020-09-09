import datetime as dt
import re

import numpy as np
import pandas as pd
import pytest
import scipy.interpolate
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

import silicone.multiple_infillers
from silicone.database_crunchers import EqualQuantileWalk

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
    tclass = EqualQuantileWalk
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
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CH4"])
        assert callable(res)

    def test_derive_relationship_with_nans(self):
        tdb = self.tdb.copy()
        tdb.loc[(tdb["variable"] == _eco2) & (tdb["model"] == _ma), 2050] = np.nan
        tcruncher = self.tclass(IamDataFrame(tdb))
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CH4"])
        # just make sure that this runs through and no error is raised
        assert callable(res)

    def test_derive_relationship_with_multicolumns(self):
        tdb = self.tdb.copy()
        tcruncher = self.tclass(IamDataFrame(tdb))
        error_msg = re.escape(
            "For `EqualQuantileWalk`, ``variable_leaders`` should only "
            "contain one variable"
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|CO2", ["Emissions|CH4", "Emissions|HFC|C5F12"],
            )

    @pytest.mark.parametrize("add_col", [None, "extra_col"])
    def test_relationship_usage(self, test_db, simple_df, add_col):
        tcruncher = self.tclass(test_db)
        lead = ["Emissions|CO2"]
        follow = "Emissions|CH4"
        simple_df = self._adjust_time_style_to_match(simple_df, test_db)
        res = tcruncher.derive_relationship(follow, lead)
        if add_col:
            add_col_val = "blah"
            simple_df[add_col] = add_col_val
            simple_df = IamDataFrame(simple_df.data)
            assert simple_df.extra_cols[0] == add_col

        infilled = res(simple_df)
        # We compare the results with the expected results: for T1, we are below the
        # lower limit on the first, in the middle on the second scenario. At later times
        # we are always above the highest value.
        time_filter = {infilled.time_col: [infilled[infilled.time_col][0]]}
        sorted_follow_t0 = np.sort(
            test_db.filter(variable=follow, **time_filter)["value"].values
        )
        assert np.allclose(
            infilled.filter(**time_filter)["value"].values,
            [sorted_follow_t0[0], sorted_follow_t0[1],],
        )
        for time_ind in range(1, 3):
            time_filter = {infilled.time_col: [infilled[infilled.time_col][time_ind]]}
            assert np.allclose(
                infilled.filter(**time_filter)["value"].values,
                max(
                    test_db.filter(variable=follow)
                    .filter(**time_filter)["value"]
                    .values
                ),
            )

        # Test we can append our answer
        append_df = simple_df.filter(variable=lead).append(infilled)
        assert append_df.filter(variable=follow).equals(infilled)

        if add_col:
            assert all(append_df[add_col] == add_col_val)

    def test_numerical_relationship(self, test_db):
        # Calculate the values using the cruncher for a fairly detailed dataset
        large_db = IamDataFrame(self.large_db.copy())
        large_db = self._adjust_time_style_to_match(large_db, test_db,)
        large_db["value"] -= 0.1
        test_db.filter(year=2010, inplace=True)
        tcruncher = self.tclass(large_db)
        lead = ["Emissions|CO2"]
        follow = "Emissions|CH4"
        res = tcruncher.derive_relationship(follow, lead)
        assert callable(res)
        to_find = test_db
        crunched = res(to_find)

        # Calculate the same values numerically
        xs = np.sort(large_db.filter(variable="Emissions|CO2")["value"].values)
        ys = np.sort(large_db.filter(variable="Emissions|CH4")["value"].values)
        quantiles_of_x = np.arange(len(xs)) / (len(xs) - 1)
        quant_of_y = scipy.interpolate.interp1d(
            xs, quantiles_of_x, bounds_error=False, fill_value=(0, 1)
        )(to_find.filter(variable=lead)["value"].values)

        expected = np.quantile(ys, quant_of_y)

        assert all(crunched["value"].values == expected)

    def test_uneven_lead_follow_len_short_lead(self, test_db):
        # In the event that there is only one set of lead data, all follow data should
        # be the same at a given time, and should equal the mean value in the input.
        test_db.filter(model=_ma, inplace=True)
        lead = ["Emissions|CH4"]
        follow = "Emissions|CO2"
        one_lead = test_db.copy()
        one_lead.data = one_lead.data.iloc[4:]
        tcruncher = self.tclass(one_lead)
        res = tcruncher.derive_relationship(follow, lead)
        infilled = res(test_db)
        assert np.allclose(
            infilled.filter(scenario=_sa)["value"]._values,
            infilled.filter(scenario=_sb)["value"]._values,
        )
        # We expect the specific values to equal the means of the two follow scenarios
        assert np.allclose(infilled.filter(scenario=_sa)["value"], [1, 2, 2.5, 2.5])

    def test_uneven_lead_timeseries(self, test_db):
        # In the event that some of the data is missing at one time, we get nans in a
        # timeseries. This checks that the calculation still works.
        test_db.filter(model=_ma, inplace=True)
        lead = ["Emissions|CH4"]
        follow = "Emissions|CO2"
        one_follow = test_db.copy()
        # We remove two of the CH4 values at the same time
        one_follow.data = one_follow.data.drop([0, 16])
        one_follow.data["scenario"] += "_orig"
        one_follow.append(test_db, inplace=True)
        one_follow = IamDataFrame(one_follow.data)
        tcruncher = self.tclass(one_follow)
        res = tcruncher.derive_relationship(follow, lead)
        infilled = res(test_db)

        assert all(~np.isnan(infilled["value"]))

    def test_with_one_value_in_infiller_db(self, test_db, caplog):
        # The calculation is different with only one entry in the infiller db, as in the
        # uneven lead case above, although this time there are no nans involved in the
        # calculation.
        one_entry_db = test_db.filter(
            scenario=test_db.scenarios()[0], model=test_db.models()[0],
        )
        tcruncher = self.tclass(one_entry_db)
        follow = "Emissions|CH4"
        lead = ["Emissions|CO2"]
        res = tcruncher.derive_relationship(follow, lead)
        infilled = res(test_db)
        assert np.allclose(
            infilled["value"],
            one_entry_db.filter(variable=follow)["value"].to_list() * 4,
        )

    def test_extreme_values_relationship(self):
        # Our cruncher has a closest-point extrapolation algorithm and therefore
        # should return the same values when filling for data outside tht limits of
        # its cruncher

        # Calculate the values using the cruncher for a fairly detailed dataset
        large_db_int = IamDataFrame(self.large_db)
        tcruncher = self.tclass(large_db_int)
        follow = "Emissions|CH4"
        lead = ["Emissions|CO2"]
        res = tcruncher.derive_relationship(follow, lead)
        crunched = res(large_db_int)

        # Increase the maximum values
        modify_extreme_db = large_db_int.filter(variable="Emissions|CO2").copy()
        max_scen = modify_extreme_db["scenario"].loc[
            modify_extreme_db["value"] == max(modify_extreme_db["value"])
        ]
        ind = modify_extreme_db["value"].idxmax
        modify_extreme_db["value"].loc[ind] += 10
        extreme_crunched = res(modify_extreme_db)
        # Check results are the same
        assert crunched.equals(extreme_crunched)
        # Also check that the results are correct
        assert crunched.filter(scenario=max_scen)["value"].iloc[0] == max(
            large_db_int.filter(variable=follow)["value"].values
        )

        # Repeat with reducing the minimum value. This works differently because the
        # minimum point is doubled. This modification causes the cruncher to pick the
        # lower value.
        min_scen = modify_extreme_db["scenario"].loc[
            modify_extreme_db["value"] == min(modify_extreme_db["value"])
        ]
        ind = modify_extreme_db["value"].idxmin
        modify_extreme_db["value"].loc[ind] -= 10
        extreme_crunched = res(modify_extreme_db)
        assert crunched.filter(scenario=min_scen)["value"].iloc[0] != min(
            large_db_int.filter(variable=follow)["value"].values
        )
        assert extreme_crunched.filter(scenario=min_scen)["value"].iloc[0] == min(
            large_db_int.filter(variable=follow)["value"].values
        )

    def test_derive_relationship_same_gas(self, test_db, test_downscale_df):
        # Given only a single data series, we recreate the original pattern
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"])
        crunched = res(test_db)
        assert np.allclose(
            crunched["value"].reset_index(drop=True),
            test_db.filter(variable="Emissions|CO2")["value"].reset_index(drop=True),
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
            tcruncher.derive_relationship("Emissions|CH4", variable_leaders)

    def test_crunch_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|CO2"]
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CH4", variable_leaders)
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
            tcruncher.derive_relationship(variable_follower, ["Emissions|CO2"])

    def test_relationship_no_data(self, test_db):
        tcruncher = self.tclass(test_db)
        follow = "Silly name"
        lead = ["other silly name"]
        err_mssg = re.escape(
            "No data for `variable_leaders` ({}) in database".format(lead)
        )
        with pytest.raises(ValueError, match=err_mssg):
            tcruncher.derive_relationship(follow, lead)

    def test_relationship_usage_wrong_unit(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"])

        exp_units = test_db.filter(variable="Emissions|CO2")["unit"].iloc[0]

        wrong_unit = "t C/yr"
        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        test_downscale_df["unit"] = wrong_unit

        error_msg = re.escape(
            "Units of lead variable is meant to be `{}`, found `{}`".format(
                exp_units, [wrong_unit]
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            res(test_downscale_df)

    def test_relationship_usage_wrong_time(self):
        tdb = IamDataFrame(self.tdb)
        tcruncher = self.tclass(tdb)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"])

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

        filler = tcruncher.derive_relationship("Emissions|CH4", ["Emissions|CO2"])

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        times_we_have = test_db.filter(year=2030, keep=False).timeseries().columns

        error_msg = re.escape(
            "Not all required timepoints are present in the database we "
            "crunched, we crunched \n\t{} for the lead and \n\t{} for the follow"
            " \nbut you passed in \n\t{}".format(
                times_we_have, times_we_have, test_db.timeseries().columns,
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)
