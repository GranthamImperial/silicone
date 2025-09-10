import logging
import re

import numpy as np
import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import TimeDepRatio

_msa = ["model_a", "scen_a"]
_msb = ["model_a", "scen_b"]


class TestDatabaseCruncherTimeDepRatio(_DataBaseCruncherTester):
    tclass = TimeDepRatio
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 2, 3],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 0.5, 1.5],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
    )
    irregular_msa = ["model_b", "scen_a"]
    unequal_df = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 1, 3],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 3],
            irregular_msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 0.5, 1.5],
            _msb + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 9, 3],
            _msb + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 3],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
    )
    tdownscale_df = pd.DataFrame(
        [
            [
                "model_b",
                "scen_b",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.25,
                2,
                3,
            ],
            [
                "model_b",
                "scen_c",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.2,
                2.3,
                2.8,
            ],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        assert callable(res)

    def test_derive_relationship_error_multiple_lead_vars(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "For `TimeDepRatio`, ``variable_leaders`` should only contain one variable"
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|HFC|C5F12", ["a", "b"])

    def test_derive_relationship_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|HFC|C2F6"]
        tcruncher = self.tclass(test_db.filter(variable=variable_leaders, keep=False))

        error_msg = re.escape(
            "No data for `variable_leaders` ({}) in database".format(variable_leaders)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|HFC|C5F12", variable_leaders)

    def test_derive_relationship_error_no_info_follower(self, test_db):
        # test that crunching fails if there's no data about the follower gas in the
        # database
        variable_follower = "Emissions|HFC|C5F12"
        tcruncher = self.tclass(test_db.filter(variable=variable_follower, keep=False))

        error_msg = re.escape(
            "No data for `variable_follower` ({}) in database".format(variable_follower)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable_follower, ["Emissions|HFC|C2F6"])

    def test_relationship_usage_not_enough_time(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        error_msg = re.escape(
            "Not all required timepoints are in the data for the lead"
            " gas (Emissions|HFC|C2F6)"
        )
        with pytest.raises(ValueError, match=error_msg):
            res = filler(test_downscale_df)

    @pytest.mark.parametrize("use_consistent", [True, False])
    def test_relationship_usage_multiple_bad_data(
        self, unequal_df, test_downscale_df, use_consistent
    ):
        tcruncher = self.tclass(unequal_df)
        error_msg = "The follower and leader data have different sizes"
        if use_consistent:
            # In this case we remove the mismatched data so there is no problem
            filler = tcruncher.derive_relationship(
                "Emissions|HFC|C5F12",
                ["Emissions|HFC|C2F6"],
                only_consistent_cases=use_consistent,
            )
        else:
            # In this case we expect the mismatched data to be problematic
            with pytest.raises(ValueError, match=error_msg):
                filler = tcruncher.derive_relationship(
                    "Emissions|HFC|C5F12",
                    ["Emissions|HFC|C2F6"],
                    only_consistent_cases=use_consistent,
                )

    @pytest.mark.parametrize(
        "match_sign,add_col",
        [(True, None), (True, "extra"), (False, None), (False, "extra")],
    )
    def test_relationship_usage_multiple_data(
        self, unequal_df, test_downscale_df, match_sign, add_col, caplog
    ):
        # quiet pyam
        caplog.set_level(logging.ERROR, logger="pyam")

        equal_df = unequal_df.filter(model="model_a")
        tcruncher = self.tclass(equal_df)
        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df, equal_df
        ).filter(year=[2010, 2015])
        lead = ["Emissions|HFC|C2F6"]
        follow = "Emissions|HFC|C5F12"
        filler = tcruncher.derive_relationship(follow, lead, match_sign)
        if add_col:
            add_col_val = "blah"
            test_downscale_df = test_downscale_df.data
            test_downscale_df[add_col] = add_col_val
            test_downscale_df = IamDataFrame(test_downscale_df)
            assert test_downscale_df.extra_cols[0] == add_col
        with caplog.at_level(logging.INFO, logger="silicone.crunchers"):
            res = filler(test_downscale_df)
        # We did not have any negative values so do not expect errors to be logged
        assert len(caplog.record_tuples) == 0
        lead_iamdf = test_downscale_df.filter(variable=lead)

        exp = lead_iamdf.timeseries()
        # The follower values are 1 and 9 (average 5), the leader values are all 1
        # hence we expect the input * 5 as output.
        exp[exp.columns[0]] = exp[exp.columns[0]] * 5
        exp = exp.reset_index()
        exp["variable"] = follow
        exp["unit"] = "kt C5F12/yr"
        exp = IamDataFrame(exp)

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            test_downscale_df.timeseries().columns.values.squeeze(),
        )

        # Test we can append our answer
        appended_df = test_downscale_df.filter(variable=lead).append(res)
        assert appended_df.filter(variable=follow).equals(res)
        if add_col:
            assert all(appended_df[add_col] == add_col_val)

    @pytest.mark.parametrize("match_sign", [True, False])
    def test_relationship_negative_specific(
        self, unequal_df, test_downscale_df, match_sign, caplog
    ):
        # quiet pyam
        caplog.set_level(logging.ERROR, logger="pyam")

        # Test that match_sign results in the correct multipliers when negative values
        # are added to the positive ones.
        follow = "Emissions|HFC|C5F12"
        lead = ["Emissions|HFC|C2F6"]
        equal_df = unequal_df.filter(model="model_a")
        invert_sign = equal_df.filter(variable=lead).data
        invert_sign["value"] = -1 * invert_sign["value"]
        invert_sign = invert_sign.append(equal_df.filter(variable=follow).data)
        invert_sign["model"] = "negative_model"
        invert_sign = IamDataFrame(invert_sign)
        equal_df.append(invert_sign, inplace=True)
        tcruncher = self.tclass(equal_df)

        # Invert the sign of the infillee too, as we haven't tested the formula for
        # negative values
        test_downscale_df = test_downscale_df.data
        test_downscale_df["value"] = -1 * test_downscale_df["value"]
        test_downscale_df = IamDataFrame(test_downscale_df)
        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df, equal_df
        ).filter(year=[2010, 2015])
        filler = tcruncher.derive_relationship(
            variable_follower=follow, variable_leaders=lead, same_sign=match_sign
        )

        with caplog.at_level(logging.INFO, logger="silicone.crunchers"):
            res = filler(test_downscale_df)
        # we expect there to be an error message for a negative result
        assert len(caplog.record_tuples) == 1
        assert caplog.record_tuples[-1][2] == (
            "Note that the lead variable {} goes negative. The time dependent "
            "ratio cruncher can produce unexpected results in this case.".format(lead)
        )
        # if we have match sign on, this is identical to the above except for -ve.
        if match_sign:
            lead_iamdf = test_downscale_df.filter(variable="Emissions|HFC|C2F6")
            exp = lead_iamdf.timeseries()
            # The follower values are 1 and 9 (average 5), the leader values
            # are all -1, hence we expect the input * -5 as output.
            exp[exp.columns[0]] = exp[exp.columns[0]] * -5
            exp[exp.columns[1]] = exp[exp.columns[1]] * -1
            exp = exp.reset_index()
            exp["variable"] = "Emissions|HFC|C5F12"
            exp["unit"] = "kt C5F12/yr"
            exp = IamDataFrame(exp)
            # Test that our constructed value is the same as the result
            pd.testing.assert_frame_equal(
                res.timeseries(), exp.timeseries(), check_like=True
            )

            # comes back on input timepoints
            np.testing.assert_array_equal(
                res.timeseries().columns.values.squeeze(),
                test_downscale_df.timeseries().columns.values.squeeze(),
            )
        else:
            # If we combine all signs together, there is no net lead but a net follow,
            # so we have have a multiplier of infinity.
            assert all(res.data["value"] == -np.inf)

    @pytest.mark.parametrize(
        "match_sign,consistent_cases",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_relationship_usage_nans(
        self, unequal_df, test_downscale_df, match_sign, caplog, consistent_cases
    ):
        leader = ["Emissions|HFC|C2F6"]
        equal_df = unequal_df.filter(model="model_a").data
        equal_df["value"].iloc[0] = np.nan
        equal_df = IamDataFrame(equal_df)
        tcruncher = self.tclass(equal_df)
        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df, equal_df
        ).filter(year=[2010, 2015])
        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12",
            leader,
            match_sign,
            only_consistent_cases=consistent_cases,
        )
        if match_sign or consistent_cases:
            res = filler(test_downscale_df)
            assert 2010 in res.data["year"].values
            # The nan'd data is ignored in these cases, so the ratio in 2010 is 1:9
            assert np.allclose(
                res.filter(year=2010).data["value"],
                test_downscale_df.filter(year=2010).data["value"] * 9,
            )
        else:
            err_msg = re.escape(
                "Attempt to infill {} data using the time_dep_ratio cruncher "
                "where the infillee data has a sign not seen in the infiller "
                "database for year "
                "{}.".format(leader, 2010)
            )
            # We have a single nan in the code, resulting in an error being thrown.
            with pytest.raises(ValueError, match=err_msg):
                filler(test_downscale_df)

    @pytest.mark.parametrize(
        "match_sign,consistent_cases",
        [(True, True), (True, False), (False, True), (False, False)],
    )
    def test_relationship_usage_consistent_cases_interacts_with_sign_match(
        self, unequal_df, test_downscale_df, match_sign, caplog, consistent_cases
    ):
        # We set the only complete timeseries to have a negative lead value in 2010.
        # If we match sign, we therefore throw an error infilling the positive value.
        # If we include inconsistent data, we throw a different error as the data
        # lengths are inconsistent.
        # Otherwise, the ratio is -1 : 1.
        leader = ["Emissions|HFC|C2F6"]
        equal_df = unequal_df.filter(scenario="scen_a").data
        equal_df["value"].iloc[0] = -1
        equal_df = IamDataFrame(equal_df)
        tcruncher = self.tclass(equal_df)
        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df, equal_df
        ).filter(year=[2010, 2015])
        if not consistent_cases:
            err_msg = re.escape("The follower and leader data have different sizes")
            with pytest.raises(ValueError, match=err_msg):
                tcruncher.derive_relationship(
                    "Emissions|HFC|C5F12",
                    leader,
                    match_sign,
                    only_consistent_cases=consistent_cases,
                )
            equal_df.filter(model="model_a", inplace=True)
            tcruncher = self.tclass(equal_df)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12",
            leader,
            match_sign,
            only_consistent_cases=consistent_cases,
        )
        if match_sign:
            err_msg = re.escape(
                "Attempt to infill {} data using the time_dep_ratio cruncher "
                "where the infillee data has a sign not seen in the infiller "
                "database for year "
                "{}.".format(leader, 2010)
            )
            # We have a single nan in the code, resulting in a warning being thrown.
            with pytest.raises(ValueError, match=err_msg):
                filler(test_downscale_df)
        else:
            res = filler(test_downscale_df)
            assert np.allclose(
                res.filter(year=2010).data["value"],
                test_downscale_df.filter(year=2010).data["value"] * -1,
            )

    @pytest.mark.parametrize(
        "match_sign, input_sign", [(True, +1), (True, -1), (False, +1), (False, -1)]
    )
    def test_relationship_usage(
        self, test_db, test_downscale_df, match_sign, input_sign, caplog
    ):
        tcruncher = self.tclass(test_db)
        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"], match_sign
        )
        test_downscale_df = (
            self._adjust_time_style_to_match(test_downscale_df, test_db)
            .filter(year=[2010, 2015])
            .data
        )
        test_downscale_df["value"] = test_downscale_df["value"] * input_sign
        test_downscale_df = IamDataFrame(test_downscale_df)
        if match_sign and input_sign < 0:
            with pytest.raises(ValueError):
                filler(test_downscale_df)
        else:
            res = filler(test_downscale_df)
            lead_iamdf = test_downscale_df.filter(variable="Emissions|HFC|C2F6")
            # We have a ratio of (2/0.5) = 4 for 2010 and (3/1.5) = 2 for 2015
            exp = lead_iamdf.timeseries()
            exp[exp.columns[0]] = exp[exp.columns[0]] * 4
            exp[exp.columns[1]] = exp[exp.columns[1]] * 2
            exp = exp.reset_index()
            exp["variable"] = "Emissions|HFC|C5F12"
            exp["unit"] = "kt C5F12/yr"
            exp = IamDataFrame(exp)

            pd.testing.assert_frame_equal(
                res.timeseries(), exp.timeseries(), check_like=True
            )

            # comes back on input timepoints
            np.testing.assert_array_equal(
                res.timeseries().columns.values.squeeze(),
                test_downscale_df.timeseries().columns.values.squeeze(),
            )

    def test_multiple_units_breaks_infillee(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = (
            self._adjust_time_style_to_match(test_downscale_df, test_db)
            .filter(year=[2010, 2015])
            .data
        )
        test_downscale_df["unit"].iloc[0] = "bad units"
        test_downscale_df = IamDataFrame(test_downscale_df)
        with pytest.raises(
            AssertionError, match="There are multiple units for the lead variable."
        ):
            res = filler(test_downscale_df)

    @pytest.mark.parametrize("consistent_cases", [True, False])
    def test_multiple_units_breaks_infiller_follower(self, test_db, consistent_cases):
        test_db = test_db.data
        test_db["unit"].iloc[2] = "bad units"
        test_db = IamDataFrame(test_db)
        if consistent_cases:
            error_str = (
                "No data is complete enough to use in the time-dependent ratio cruncher"
            )
        else:
            error_str = "There are multiple/no units in follower data"

        with pytest.raises(ValueError, match=error_str):
            tcruncher = self.tclass(test_db)
            tcruncher.derive_relationship(
                "Emissions|HFC|C5F12",
                ["Emissions|HFC|C2F6"],
                only_consistent_cases=consistent_cases,
            )

    @pytest.mark.parametrize("consistent_cases", [True, False])
    def test_multiple_units_breaks_infiller_leader(self, test_db, consistent_cases):
        test_db = test_db.data
        test_db["unit"].iloc[0] = "bad units"
        test_db = IamDataFrame(test_db)
        if consistent_cases:
            error_str = (
                "No data is complete enough to use in the time-dependent ratio cruncher"
            )
        else:
            error_str = "There are multiple/no units for the leader data."
        with pytest.raises(ValueError, match=error_str):
            tcruncher = self.tclass(test_db)
            tcruncher.derive_relationship(
                "Emissions|HFC|C5F12",
                ["Emissions|HFC|C2F6"],
                only_consistent_cases=consistent_cases,
            )
