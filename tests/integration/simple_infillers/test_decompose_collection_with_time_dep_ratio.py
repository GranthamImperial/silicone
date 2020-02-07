import re

import numpy as np
import pandas as pd
import pyam
import pytest
from silicone.utils import convert_units_to_MtCO2_equiv, _adjust_time_style_to_match

from silicone.simple_infillers.decompose_collection_with_time_dep_ratio import (
    DecomposeCollectionTimeDepRatio,
)

_msa = ["model_a", "scen_a"]
_msb = ["model_a", "scen_b"]


class TestGasDecomposeTimeDepRatio:
    tclass = DecomposeCollectionTimeDepRatio
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

    def test__construct_consistent_values(self, test_db):
        test_db_co2 = convert_units_to_MtCO2_equiv(test_db)
        tcruncher = self.tclass(test_db_co2)
        aggregate_name = "agg"
        assert aggregate_name not in tcruncher._db.variables().values
        component_ratio = ["Emissions|HFC|C2F6", "Emissions|HFC|C5F12"]
        consistent_vals = tcruncher._construct_consistent_values(
            aggregate_name, component_ratio, test_db_co2
        )
        assert aggregate_name in consistent_vals["variable"].values
        consistent_vals = pyam.IamDataFrame(consistent_vals).timeseries()
        timeseries_data = tcruncher._db.timeseries()
        assert all(
            [
                np.allclose(
                    consistent_vals.iloc[0].iloc[ind],
                    timeseries_data.iloc[0].iloc[ind]
                    + timeseries_data.iloc[1].iloc[ind],
                )
                for ind in range(len(timeseries_data.iloc[0]))
            ]
        )

    def test_infill_components_error_no_lead_vars(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "The database to infill does not have the aggregate variable"
        )
        with pytest.raises(AssertionError, match=error_msg):
            tcruncher.infill_components("c", ["a", "b"], test_db)

    def test_db_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|HFC|C2F6"]
        follower = "Emissions|HFC|C5F12"
        tcruncher = self.tclass(test_db.filter(variable=variable_leaders, keep=False))

        error_msg = re.escape(
            "Attempting to construct a consistent {} but none of the components "
            "present".format(follower)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.infill_components(
                follower,
                variable_leaders,
                test_db.filter(variable=variable_leaders, keep=False),
            )

    def test_db_error_preexisting_follow_data(self, test_db):
        # test that crunching fails if there's no data about the follower gas in the
        # database
        variable_follower = "Emissions|HFC|C5F12"
        variable_leaders = ["Emissions|HFC|C2F6"]
        tcruncher = self.tclass(test_db)

        error_msg = re.escape(
            "The database to infill already has some component variables"
        )
        with pytest.raises(AssertionError, match=error_msg):
            tcruncher.infill_components(variable_follower, variable_leaders, test_db)

    def test_db_error_multiple_units(self, test_db):
        # test that crunching fails if there's no data about the follower gas in the
        # database
        aggregate_name = "Emissions|HFC|C5F12"
        components = ["Emissions|HFC|C2F6"]
        test_db.data["variable"] = components[0]
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "Too many units found to make a consistent {}".format(aggregate_name)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher._construct_consistent_values(aggregate_name, components, test_db)

    def test_relationship_usage_not_enough_time(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        test_downscale_df = _adjust_time_style_to_match(test_downscale_df, test_db)
        error_msg = re.escape(
            "Not all required timepoints are in the data for the lead"
            " gas (Emissions|HFC|C2F6)"
        )
        with pytest.raises(ValueError, match=error_msg):
            filler = tcruncher.infill_components(
                "Emissions|HFC|C2F6", ["Emissions|HFC|C5F12"], test_downscale_df
            )

    def test_relationship_usage_works(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        test_downscale_df = _adjust_time_style_to_match(test_downscale_df, test_db)
        if test_db.time_col == "year":
            test_downscale_df.filter(
                year=test_db.data[test_db.time_col].values, inplace=True
            )
        else:
            test_downscale_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        components = ["Emissions|HFC|C5F12"]
        filled = tcruncher.infill_components(
            "Emissions|HFC|C2F6", components, test_downscale_df
        )
        # The value returned should include only one entry with
        assert len(filled) == 1
        assert all(y == components[0] for y in filled[0].variables())
        assert np.allclose(filled[0].data["value"], test_downscale_df.data["value"])
