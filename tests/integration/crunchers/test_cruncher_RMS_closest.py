import re

import numpy as np
import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import DatabaseCruncherRMSClosest


class TestDatabaseCruncherRMSClosest(_DataBaseCruncherTester):
    tclass = DatabaseCruncherRMSClosest
    tdownscale_df = pd.DataFrame(
        [
            ["model_b", "scen_b", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 2, 3],
            ["model_b", "scen_c", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1.1, 2.2, 2.8],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        assert isinstance(res, object)

    def test_derive_relationship_error_multiple_lead_vars(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "For `DatabaseCruncherRMSClosest`, ``variable_leaders`` should only "
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
                [["ma", "sb", "World", "Emissions|HFC|C5F12", "kt C5F12/yr", 1, 2]],
                columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
            ),
            pd.DataFrame(
                [
                    ["ma", "sa", "World", "Emissions|HFC|C5F12", "kt C5F12/yr", 1],
                    ["ma", "sb", "World", "Emissions|HFC|C5F12", "kt C5F12/yr", 3],
                ],
                columns=["model", "scenario", "region", "variable", "unit", 2010],
            ),
        ),
    )
    def test_derive_relationship_error_too_much_info(self, test_db, extra_info):
        # test that crunching fails if there's more than a single point (whether year
        # or scenario) for the gas to downscale to in the database
        tdb = test_db.filter(variable="Emissions|HFC|C5F12", keep=False)
        tcruncher = self.tclass(
            self._join_iamdfs_time_wrangle(tdb, IamDataFrame(extra_info))
        )
        error_msg = re.escape(
            "More than one data point for `variable_follower` ({}) in database".format(
                "Emissions|HFC|C5F12"
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"])

    def test_relationship_usage(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        res = filler(test_downscale_df)

        lead_iamdf = test_downscale_df.filter(variable="Emissions|HFC|C2F6")
        lead_val_2015 = lead_iamdf.filter(year=2015).timeseries().values.squeeze()

        exp = (lead_iamdf.timeseries().T * 3.14 / lead_val_2015).T
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

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_relationship_usage_interpolation(
        self, test_db, test_downscale_df, interpolate
    ):
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df.filter(year=2015, keep=False), test_db)

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
