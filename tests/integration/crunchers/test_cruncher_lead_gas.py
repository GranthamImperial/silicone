import re

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from silicone.database_crunchers import DatabaseCruncherLeadGas

from base import _DataBaseCruncherTester


class TestDatabaseCruncherLeadGas(_DataBaseCruncherTester):
    tclass = DatabaseCruncherLeadGas
    tdownscale_df = pd.DataFrame(
        [["model_b", "scen_b", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 2, 3]],
        columns=["model", "scenario", "region", "variable", "unit", 2015, 2020, 2050],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|HFC|C5F12", "Emissions|HFC|C2F6")
        assert isinstance(res, object)
        assert False, "sjkalfd"

    def test_derive_relationship_error_no_info(self, test_db):
        # TODO: move this to base testing class
        # test that crunching fails if there's no data about the lead gas in the
        # database
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|HFC|C5F12", "Emissions|HFC|C2F6")
        assert isinstance(res, object)
        assert False, "sjkalfd"

    def test_derive_relationship_error_too_much_info(self, test_db):
        # test that crunching fails if there's more than a single point (whether year
        # or scenario) for the gas to downscale to in the database
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|HFC|C5F12", "Emissions|HFC|C2F6")
        assert isinstance(res, object)
        assert False, "sjkalfd"

    def test_relationship_usage(self, test_db, test_downscale_df):
        # TODO, make this an abstract method in base class to ensure all crunchers
        # have at least one basic test of how their output works
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", "Emissions|HFC|C2F6"
        )
        res = filler(test_downscale_df)

        res_downscaled = res.filter(
            variable="Emissions|HFC|C5F12", region="World", unit="kt C5F12/yr"
        )
        res_downscaled_timeseries = res_downscaled.timeseries()

        lead_iamdf = test_downscale_df.filter(
            variable="Emissions|HFC|C2F6", region="World", unit="kt C2F6/yr"
        )
        lead_vals = lead_iamdf.timeseries().values.squeeze()
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

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_relationship_usage_interpolation(
        self, test_db, test_downscale_df, interpolate
    ):
        # TODO, make this an abstract method in base class to ensure all crunchers
        # have at least one basic test of how their output works
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", "Emissions|HFC|C2F6"
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

    def check_downscaled_variable_metadata(self, downscaled_iamdf):
        assert downscaled_iamdf.models().tolist() == ["model_b"]
        assert downscaled_iamdf.scenarios().tolist() == ["scen_b"]
        assert downscaled_iamdf.regions().tolist() == ["World"]
        pd.testing.assert_frame_equal(
            downscaled_iamdf.variables(True),
            pd.DataFrame(
                ["Emissions|HFC|C5F12", "kt C5F12/yr"], columns=["variable", "unit"]
            ),
        )

    def check_downscaled_variables(self, downscaled_iamdf):
        # old variables part of res too
        pd.testing.assert_frame_equal(
            downscaled_iamdf.variables(True),
            pd.DataFrame(
                [
                    ["Emissions|HFC|C2F6", "kt C2F6/yr"],
                    ["Emissions|HFC|C5F12", "kt C5F12/yr"],
                ],
                columns=["variable", "unit"],
            ),
        )
