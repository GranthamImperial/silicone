import re

import numpy as np
import pandas as pd
import pyam
import pytest

from silicone.multiple_infillers import (
    DecomposeCollectionTimeDepRatio,
    infill_composite_values,
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
            "meta1",
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
        test_db = test_db.data
        test_db["unit"] = "kt C2F6-equiv/yr"
        test_db = pyam.IamDataFrame(test_db)
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
        test_db = test_db.data
        test_db["unit"] = "kt C2F6-equiv/yr"
        test_db = pyam.IamDataFrame(test_db)
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
        assert all(y == components[0] for y in filled.variable)
        assert np.allclose(filled.data["value"], test_downscale_df.data["value"])

    def test_relationship_usage_works_inconsistent_data(self, test_db, unequal_df):
        # Test that the decomposer function ignores scenarios with missing data when
        # performing infilling
        components = ["Emissions|HFC|C5F12", "Emissions|HFC|C2F6"]
        aggregate = "Emissions|HFC"
        to_infill = infill_composite_values(
            convert_units_to_MtCO2_equiv(test_db), composite_dic={aggregate: components}
        )
        unequal_df = _adjust_time_style_to_match(unequal_df, test_db)
        equal_df = unequal_df.filter(model="model_b", scenario="scen_a", keep=False)
        uneq_results = self.tclass(unequal_df).infill_components(
            aggregate, components, to_infill
        )
        eq_results = self.tclass(equal_df).infill_components(
            aggregate, components, to_infill
        )
        assert uneq_results.equals(eq_results)
        # Now ensure that removing data at one point in time removes that data at all
        # times
        unequal_df.filter(
            variable=components[0],
            scenario="scen_a",
            keep=False,
            inplace=True,
            **{unequal_df.time_col: unequal_df[unequal_df.time_col].iloc[0]},
        )
        equal_df.filter(
            variable=components[0], scenario="scen_a", keep=False, inplace=True,
        )
        uneq_results = self.tclass(unequal_df).infill_components(
            aggregate, components, to_infill
        )
        eq_results = self.tclass(equal_df).infill_components(
            aggregate, components, to_infill
        )
        assert uneq_results.equals(eq_results)
        # However data from later times should still create a difference if we use the
        # inconsistent cases too.
        uneq_results = self.tclass(unequal_df).infill_components(
            aggregate, components, to_infill, only_consistent_cases=False
        )
        eq_results_cons = self.tclass(equal_df).infill_components(
            aggregate, components, to_infill, only_consistent_cases=False
        )
        # Nothing changes for consistent data, eveything does for inconsistent data
        assert eq_results_cons.equals(eq_results)
        assert not eq_results_cons.equals(uneq_results)

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
        test_downscale_df = test_downscale_df.data
        test_downscale_df["variable"] = aggregate
        test_downscale_df = pyam.IamDataFrame(test_downscale_df)
        tcruncher = self.tclass(test_db)
        with pytest.raises(ValueError):
            filled = tcruncher.infill_components(
                aggregate, components, test_downscale_df
            )
        test_downscale_df = convert_units_to_MtCO2_equiv(test_downscale_df)
        filled = tcruncher.infill_components(aggregate, components, test_downscale_df)
        # The value returned should be a dataframe with 2 entries per original entry (4)
        assert len(filled.data) == 8
        assert all(y in filled.variable for y in components)
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
        test_db = test_db.data
        test_db["variable"] = aggregate
        test_db = pyam.IamDataFrame(test_db)
        # larger_df has an extra column, "meta1"
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

    def test_relationship_works_with_additional_cols(self, larger_df, test_db):
        # Try adding another column of data and the data is still found in the output
        aggregate = "Emissions|KyotoTotal"
        test_db = test_db.data.iloc[:2]
        test_db["variable"] = aggregate
        test_db["unit"] = "Mt CO2/yr"
        test_db["meta1"] = "some_meta"
        test_db = pyam.IamDataFrame(test_db)
        larger_df = _adjust_time_style_to_match(larger_df, test_db)
        tcruncher = self.tclass(larger_df)
        if test_db.time_col == "year":
            larger_df.filter(year=test_db.data[test_db.time_col].values, inplace=True)
        else:
            larger_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        components = ["Emissions|CH4", "Emissions|CO2"]
        returned = tcruncher.infill_components(aggregate, components, test_db)
        assert len(returned.data) == 2 * len(test_db.data)
        assert returned.extra_cols == test_db.extra_cols
