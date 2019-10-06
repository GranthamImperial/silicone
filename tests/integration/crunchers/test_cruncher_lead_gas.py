import datetime as dt
import re

import numpy as np
import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import DatabaseCruncherLeadGas

# test that using filler with `time` time_col if you derived the relationship with `year` time_col explodes


class TestDatabaseCruncherLeadGas(_DataBaseCruncherTester):
    tclass = DatabaseCruncherLeadGas
    tdownscale_df = pd.DataFrame(
        [["model_b", "scen_b", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 2, 3]],
        columns=["model", "scenario", "region", "variable", "unit", 2015, 2020, 2050],
    )

    def _join_iamdfs_time_wrangle(self, base, other):
        # TODO: put this in base class
        return base.append(self._adjust_time_style_to_match(other, base))

    def _adjust_time_style_to_match(self, in_df, target_df):
        # TODO: put this in base class
        if in_df.time_col != target_df.time_col:
            in_df = in_df.timeseries()
            if target_df.time_col == "time":
                target_df_year_map = {v.year: v for v in target_df.timeseries().columns}
                in_df.columns = in_df.columns.map(
                    lambda x: target_df_year_map[x]
                    if x in target_df_year_map
                    else dt.datetime(x, 1, 1)
                )
            else:
                in_df.columns = in_df.columns.map(lambda x: x.year)
            return IamDataFrame(in_df)

        return in_df

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        assert isinstance(res, object)

    def test_derive_relationship_error_no_info(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|HFC|C2F6"]
        tcruncher = self.tclass(test_db.filter(variable=variable_leaders, keep=False))

        error_msg = re.escape(
            "No data for `variable_leaders` ({}) in database".format(variable_leaders)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|HFC|C5F12", variable_leaders)

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
        # TODO, make this an abstract method in base class to ensure all crunchers
        # have at least one basic test of how their output works
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        res = filler(test_downscale_df)

        lead_iamdf = test_downscale_df.filter(variable="Emissions|HFC|C2F6")
        lead_val_2015 = lead_iamdf.filter(year=2015).timeseries().values.squeeze()

        exp = lead_iamdf.timeseries() * 3.14 / lead_val_2015
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
        # TODO, make this an abstract method in base class to ensure all crunchers
        # have at least one basic test of how their output works
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        if not interpolate:
            error_msg = re.escape(
                "Required downscaling year (2015) is not in the data for the "
                "lead gas (Emissions|HFC|C2F6)"
            )
            with pytest.raises(ValueError, match=error_msg):
                filler(test_downscale_df, interpolate=interpolate)
            return

        res = filler(test_downscale_df, interpolate=interpolate)

        res_downscaled = res.filter(
            variable="Emissions|HFC|C5F12", region="World", unit="kt C5F12/yr"
        )
        res_downscaled_timeseries = res_downscaled.timeseries()

        lead_iamdf = test_downscale_df.filter(
            variable="Emissions|HFC|C2F6", region="World", unit="kt C2F6/yr"
        )
        lead_vals = lead_iamdf.timeseries().values.squeeze()
        lead_iamdf.interpolate(2015)
        lead_val_2015 = lead_iamdf.filter(year=2015).timeseries().values.squeeze()

        np.testing.assert_array_equal(
            res_downscaled_timeseries.values.squeeze(), lead_vals * 3.14 / lead_val_2015
        )
        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            test_downscale_df.timeseries().columns.values.squeeze(),
        )

        self.check_downscaled_variables(res)
        self.check_downscaled_variable_metadata(res_downscaled)
