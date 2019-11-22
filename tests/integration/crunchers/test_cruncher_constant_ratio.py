import logging
import re

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from silicone.database_crunchers import DatabaseCruncherConstantRatio

_msa = ["model_a", "scen_a"]
_msb = ["model_a", "scen_b"]


class TestDatabaseCruncherTimeDepRatio:
    tclass = DatabaseCruncherConstantRatio
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

    def test_init_with_db(self, test_db, caplog):
        with caplog.at_level(
            logging.INFO, logger="silicone.database_crunchers.constant_ratio"
        ):
            self.tclass(test_db)

        assert caplog.record_tuples == [(
            "silicone.database_crunchers.constant_ratio",  # namespacing
            logging.INFO,  # level
            "{} won't use any information from the database".format(self.tclass),  # message
        )]

    def test_derive_relationship(self):
        tcruncher = self.tclass()
        res = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"], ratio=0.5, units="Some_unit"
        )
        assert callable(res)

    def test_derive_relationship_error_multiple_lead_vars(self):
        tcruncher = self.tclass()
        error_msg = re.escape(
            "For `DatabaseCruncherConstantRatio`, ``variable_leaders`` should only "
            "contain one variable"
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|HFC|C5F12", ["a", "b"], ratio=0.5, units="Some_unit"
            )

    def test_relationship_usage(self, unequal_df, test_downscale_df):
        equal_df = unequal_df.filter(model="model_a")
        units = "new units"
        tcruncher = self.tclass()
        test_downscale_df = test_downscale_df.filter(year=[2010, 2015])
        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"], ratio=2, units=units
        )
        res = filler(test_downscale_df)

        exp = test_downscale_df.filter(variable="Emissions|HFC|C2F6")
        exp.data["variable"] = "Emissions|HFC|C5F12"
        exp.data["value"] = exp.data["value"] * 2
        exp.data["unit"] = units

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            test_downscale_df.timeseries().columns.values.squeeze(),
        )

    def test_relationship_usage_set_0(self, test_downscale_df):
        tcruncher = self.tclass()

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"], ratio=0, units="units"
        )

        test_downscale_df = test_downscale_df.filter(year=[2010, 2015])
        res = filler(test_downscale_df)

        exp = test_downscale_df.filter(variable="Emissions|HFC|C2F6").data

        exp["value"] = 0
        exp["variable"] = "Emissions|HFC|C5F12"
        exp["unit"] = "units"
        exp = IamDataFrame(exp)

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            test_downscale_df.timeseries().columns.values.squeeze(),
        )

    def test_multiple_units_breaks_infillee(self, test_downscale_df):
        tcruncher = self.tclass()

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"], ratio=0, units="units"
        )

        test_downscale_df = test_downscale_df.filter(year=[2010, 2015])
        test_downscale_df["unit"].iloc[0] = "bad units"
        with pytest.raises(
            AssertionError, match="There are multiple units for the lead variable."
        ):
            res = filler(test_downscale_df)