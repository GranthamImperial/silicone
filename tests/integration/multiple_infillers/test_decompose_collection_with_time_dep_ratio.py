import re

import numpy as np
import pandas as pd
import pyam
import pytest

from silicone.multiple_infillers.decompose_collection_with_time_dep_ratio import (
    DecomposeCollectionTimeDepRatio,
)
from silicone.utils import _adjust_time_style_to_match, convert_units_to_MtCO2_equiv

_msa = ["model_a", "scen_a"]
_msb = ["model_a", "scen_b"]


class TestGasDecomposeTimeDepRatio:
    tclass = DecomposeCollectionTimeDepRatio
    # tdb will generate test_db
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
    larger_df = pd.DataFrame(
        [
            ["model_C", "scen_C", "World", "Emissions|BC", "Mt BC/yr", "", 1.0, 1, 1],
            ["model_C", "scen_C", "World", "Emissions|CH4", "Mt CH4/yr", "", 1.0, 1, 1],
            ["model_D", "scen_C", "World", "Emissions|CO2", "Mt CO2/yr", "", 2, 2, 2],
            ["model_D", "scen_C", "World", "Emissions|CH4", "Mt CH4/yr", "", 2, 2, 2],
            ["model_D", "scen_F", "World", "Emissions|CO2", "Mt CO2/yr", "", 4, 4, 4],
            ["model_D", "scen_F", "World", "Emissions|CH4", "Mt CH4/yr", "", 2, 2, 2],
        ],
        columns=[
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
            "meta",
            2010,
            2015,
            2050,
        ],
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

    def test_relationship_usage_not_enough_time(self, test_db, test_downscale_df):
        # Ensure that the process fails if not all times have data
        test_db.data["unit"] = "kt C2F6-equiv/yr"
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
        # Test that we get the correct results when everything is in order.
        # First fix the units problem
        test_db.data["unit"] = "kt C2F6-equiv/yr"
        tcruncher = self.tclass(test_db)
        # Fix times to agree
        test_downscale_df = _adjust_time_style_to_match(test_downscale_df, test_db)
        if test_db.time_col == "year":
            test_downscale_df.filter(
                year=test_db.data[test_db.time_col].values, inplace=True
            )
        else:
            test_downscale_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        components = ["Emissions|HFC|C5F12"]
        # Perform the calculation
        filled = tcruncher.infill_components(
            "Emissions|HFC|C2F6", components, test_downscale_df
        )
        # The values returned should include only 1 entry per input entry, since there
        # is a single input component
        assert len(filled.data) == 4
        assert all(y == components[0] for y in filled.variables())
        assert np.allclose(filled.data["value"], test_downscale_df.data["value"])

    def test_relationship_usage_works_multiple(self, test_db, test_downscale_df):
        # Test that the decomposer function works for slightly more complicated data
        # (two components).
        # Get matching times
        test_downscale_df = _adjust_time_style_to_match(test_downscale_df, test_db)
        if test_db.time_col == "year":
            test_downscale_df.filter(
                year=test_db.data[test_db.time_col].values, inplace=True
            )
        else:
            test_downscale_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        # Make the variables work for our case
        components = ["Emissions|HFC|C5F12", "Emissions|HFC|C2F6"]
        aggregate = "Emissions|HFC"
        test_downscale_df.data["variable"] = aggregate
        tcruncher = self.tclass(test_db)
        with pytest.raises(ValueError):
            filled = tcruncher.infill_components(
                aggregate, components, test_downscale_df
            )
        test_downscale_df = convert_units_to_MtCO2_equiv(test_downscale_df)
        filled = tcruncher.infill_components(aggregate, components, test_downscale_df)
        # The value returned should be a dataframe with 2 entries per original entry (4)
        assert len(filled.data) == 8
        assert all(y in filled.variables().values for y in components)
        # We also expect the amount of the variables to be conserved
        if test_db.time_col == "year":
            assert np.allclose(
                test_downscale_df.data.groupby("year").sum()["value"].values,
                convert_units_to_MtCO2_equiv(filled)
                .data.groupby("year")
                .sum()["value"]
                .values,
            )
        else:
            assert np.allclose(
                test_downscale_df.data.groupby("time").sum()["value"].values,
                convert_units_to_MtCO2_equiv(filled)
                .data.groupby("time")
                .sum()["value"]
                .values,
            )

    def test_relationship_rejects_inconsistent_columns(self, larger_df, test_db):
        # There are optional extra columns on the DataFrame objects. This test ensures
        # that an error is thrown if we add together different sorts of DataFrame.
        aggregate = "Emissions|KyotoTotal"
        test_db.data["variable"] = aggregate
        # larger_df has an extra column, "meta"
        larger_df = _adjust_time_style_to_match(larger_df, test_db)
        tcruncher = self.tclass(larger_df)
        if test_db.time_col == "year":
            larger_df.filter(year=test_db.data[test_db.time_col].values, inplace=True)
        else:
            larger_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        components = ["Emissions|CH4", "Emissions|CO2"]
        err_msg = re.escape(
            "The database and to_infill_db fed into this have inconsistent "
            "columns, which will prevent adding the data together properly."
        )
        with pytest.raises(AssertionError, match=err_msg):
            tcruncher.infill_components(aggregate, components, test_db)

    def test_relationship_ignores_incomplete_data(self, larger_df, test_db):
        # If we make the data inconsistent, we still get a consistent (if arbitrary)
        # output.
        aggregate = "Emissions|KyotoTotal"
        # This makes the data contain duplicates:
        test_db.data["variable"] = aggregate
        test_db.data["unit"] = "Mt CO2/yr"
        # We remove the extra column from the larger_df as it's not found in test_df
        larger_df.data.drop("meta", axis=1, inplace=True)
        larger_df = pyam.IamDataFrame(larger_df.data)
        larger_df = _adjust_time_style_to_match(larger_df, test_db)
        tcruncher = self.tclass(larger_df)
        if test_db.time_col == "year":
            larger_df.filter(year=test_db.data[test_db.time_col].values, inplace=True)
        else:
            larger_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        components = ["Emissions|CH4", "Emissions|CO2"]
        returned = tcruncher.infill_components(aggregate, components, test_db)
        assert len(returned.data) == len(test_db.data)
        # Make the data consistent:
        test_db.data = test_db.data.iloc[0:2]
        returned = tcruncher.infill_components(aggregate, components, test_db)
        assert len(returned.data) == 2 * len(test_db.data)
