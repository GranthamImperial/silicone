import re

import numpy as np
import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import DatabaseCruncherFixedQuantile


class TestDatabaseCruncherQuantileMatcher(_DataBaseCruncherTester):
    tclass = DatabaseCruncherFixedQuantile
    tdownscale_df = pd.DataFrame(
        [
            ["model_b", "scen_b", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 2, 3],
            ["model_b", "scen_c", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1.1, 2.2, 2.8],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db.filter(variable="Emissions|HFC|C2F6"))
        res = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        assert isinstance(res, object)

    def test_derive_relationship_error_follow_vars_already_exist(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = "There should be no pre-existing data for `variable_follower` ({}) in database".format(
            variable_follower
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship( "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"])

    def test_derive_relationship_error_multiple_lead_vars(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "For `DatabaseCruncherLeadGas`, ``variable_leaders`` should only "
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

