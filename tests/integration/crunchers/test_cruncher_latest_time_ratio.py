import re

import numpy as np
import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import LatestTimeRatio

_msa = ["model_a", "scen_a"]


class TestDatabaseCruncherLatestTimeRatio(_DataBaseCruncherTester):
    tclass = LatestTimeRatio
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", "", np.nan, 3.14],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", "", 1.2, 1.5],
        ],
        columns=["model", "scenario", "region", "variable", "unit", "meta", 2010, 2015],
    )
    tdownscale_df = pd.DataFrame(
        [
            ["model_b", "scen_b", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 2, 3],
            [
                "model_b",
                "scen_c",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.1,
                2.2,
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
            "For `LatestTimeRatio`, ``variable_leaders`` should only "
            "contain one variable"
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

    @pytest.mark.parametrize(
        "extra_info",
        (
            pd.DataFrame(
                [["ma", "sb", "World", "Emissions|HFC|C5F12", "kt C5F12/yr", "", 5, 2]],
                columns=[
                    "model",
                    "scenario",
                    "region",
                    "variable",
                    "unit",
                    "meta",
                    2010,
                    2015,
                ],
            ),
            pd.DataFrame(
                [
                    ["ma", "sa", "World", "Emissions|HFC|C5F12", "kt C5F12/yr", "", 1],
                    ["ma", "sb", "World", "Emissions|HFC|C5F12", "kt C5F12/yr", "", 3],
                ],
                columns=[
                    "model",
                    "scenario",
                    "region",
                    "variable",
                    "unit",
                    "meta",
                    2015,
                ],
            ),
        ),
    )
    def test_derive_relationship_averaging_info(self, test_db, extra_info):
        # test that crunching uses average values if there's more than a single point
        # in the latest year for the lead gas in the database
        variable_follower = "Emissions|HFC|C5F12"
        variable_leader = ["Emissions|HFC|C2F6"]
        tdb = test_db.filter(variable=variable_follower, keep=False)
        tcruncher = self.tclass(
            self._join_iamdfs_time_wrangle(tdb, IamDataFrame(extra_info))
        )
        cruncher = tcruncher.derive_relationship(variable_follower, variable_leader)
        lead_db = test_db.filter(variable=variable_leader)
        infilled = cruncher(lead_db)
        # In both cases, the average follower value at the latest time is 2. We divide
        # by the value in 2015, which we have data for in both cases.
        lead_db_time = lead_db.data[lead_db.time_col]
        latest_time = lead_db_time == max(lead_db_time)
        expected = (
            2 * lead_db.data["value"] / lead_db.data["value"].loc[latest_time].values
        )
        assert np.allclose(infilled.data["value"], expected)
        # Test that the result can be appended without problems.
        lead_db.append(infilled, inplace=True)
        assert lead_db.filter(variable=variable_follower).equals(infilled)

    def test_derive_relationship_error_no_info(self, test_db):
        # test that crunching fails if there's no data for the gas to downscale to in
        # the database
        tdb = test_db.filter(variable="Emissions|HFC|C5F12", keep=False)
        tcruncher = self.tclass(tdb)
        variable_follower = "Emissions|HFC|C5F12"
        error_msg = re.escape(
            "No data for `variable_follower` ({}) in database".format(variable_follower)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable_follower, ["Emissions|HFC|C2F6"])

    @pytest.mark.parametrize("add_col", [None, "extra_col"])
    def test_relationship_usage(self, test_db, test_downscale_df, add_col):
        tcruncher = self.tclass(test_db)
        lead = ["Emissions|HFC|C2F6"]
        follow = "Emissions|HFC|C5F12"
        filler = tcruncher.derive_relationship(follow, lead)
        if add_col:
            add_col_val = "blah"
            test_downscale_df[add_col] = add_col_val
            test_downscale_df = IamDataFrame(test_downscale_df.data)
            assert test_downscale_df.extra_cols[0] == add_col
        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        res = filler(test_downscale_df)

        lead_iamdf = test_downscale_df.filter(variable=lead)
        lead_val_2015 = lead_iamdf.filter(year=2015).timeseries().values.squeeze()

        exp = (lead_iamdf.timeseries().T * 3.14 / lead_val_2015).T
        exp = exp.reset_index()
        exp["variable"] = follow
        exp["unit"] = "kt C5F12/yr"
        if add_col:
            exp[add_col] = add_col_val
        exp = IamDataFrame(exp)

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            test_downscale_df.timeseries().columns.values.squeeze(),
        )

        # Test that we can append the output to the input
        append_df = test_downscale_df.filter(variable=lead).append(res)
        assert append_df.filter(variable=follow).equals(res)
        if add_col:
            assert all(append_df[add_col] == add_col_val)

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_relationship_usage_interpolation(
        self, test_db, test_downscale_df, interpolate
    ):
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df.filter(year=2015, keep=False), test_db
        )

        required_timepoint = test_db.filter(year=2015).data[test_db.time_col].iloc[0]
        if not interpolate:
            if isinstance(required_timepoint, pd.Timestamp):
                required_timepoint = required_timepoint.to_pydatetime()
            error_msg = re.escape(
                "Required downscaling timepoint ({}) is not in the data for the "
                "lead gas (Emissions|HFC|C2F6)".format(required_timepoint)
            )
            with pytest.raises(ValueError, match=error_msg):
                filler(test_downscale_df, interpolate=interpolate)
            return

        res = filler(test_downscale_df, interpolate=interpolate)

        lead_iamdf = test_downscale_df.filter(
            variable="Emissions|HFC|C2F6", region="World", unit="kt C2F6/yr"
        )
        exp = lead_iamdf.timeseries()

        # will have to make this more intelligent for time handling
        lead_df = lead_iamdf.timeseries()
        lead_df[required_timepoint] = np.nan
        lead_df = lead_df.reindex(sorted(lead_df.columns), axis=1)
        lead_df = lead_df.interpolate(method="index", axis=1)
        lead_val_2015 = lead_df[required_timepoint]

        exp = (exp.T * 3.14 / lead_val_2015).T.reset_index()
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
