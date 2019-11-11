import re

import numpy as np
import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import DatabaseCruncherTimeDepRatio

_msa = ["model_a", "scen_a"]


class TestDatabaseCruncherTimeDepRatio(_DataBaseCruncherTester):
    tclass = DatabaseCruncherTimeDepRatio
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 2, 3],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 0.5, 1.5],
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
            "For `DatabaseCruncherTimeDepRatio`, ``variable_leaders`` should only "
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

    def test_relationship_usage(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df, test_db
        ).filter(year=[2010, 2015])
        res = filler(test_downscale_df)

        lead_iamdf = test_downscale_df.filter(variable="Emissions|HFC|C2F6")

        exp = lead_iamdf.timeseries()
        exp[exp.columns[0]] = [5, 4.8]
        exp[exp.columns[1]] = [4, 4.6]
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

        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df, test_db
        ).filter(year=[2010, 2015])
        test_downscale_df["unit"].iloc[0] = "bad units"
        with pytest.raises(
                AssertionError,
                match="There are multiple units for the variable to infill."
        ):
            res = filler(test_downscale_df)

    def test_multiple_units_breaks_infiller_follower(self, test_db, test_downscale_df):
        test_db["unit"].iloc[2] = "bad units"
        with pytest.raises(
                ValueError,
                match="There are multiple/no units in follower data"
        ):
            tcruncher = self.tclass(test_db)
            filler = tcruncher.derive_relationship(
                "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
            )

    def test_multiple_units_breaks_infiller_leader(self, test_db, test_downscale_df):
        test_db["unit"].iloc[0] = "bad units"
        with pytest.raises(
                ValueError,
                match="There are multiple/no units for the leader data."
        ):
            tcruncher = self.tclass(test_db)
            filler = tcruncher.derive_relationship(
                "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
            )
