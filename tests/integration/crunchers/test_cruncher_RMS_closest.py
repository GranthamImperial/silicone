import re

import numpy as np
import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import concat

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

    def test_relationship_usage(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        res = filler(test_downscale_df)

        scen_b_df = test_db.filter(variable="Emissions|HFC|C5F12")
        scen_c_df = test_db.filter(variable="Emissions|HFC|C5F12")
        scen_b_df['model'] = 'model_b'
        scen_c_df['model'] = 'model_b'
        scen_b_df['scenario'] = 'scen_b'
        scen_c_df['scenario'] = 'scen_c'
        exp = concat([scen_b_df, scen_c_df])

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            exp.timeseries().columns.values.squeeze(),
        )

    def test_relationship_usage_no_overlap(
        self, test_db, test_downscale_df
    ):
        tcruncher = self.tclass(test_db.filter(year=2015))

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df.filter(year=2015, keep=False), test_db)

        error_msg = "No time series overlap between the original and unfilled data."
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)