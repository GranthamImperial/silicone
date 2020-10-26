import logging
import re

import numpy as np
import pandas as pd
import pyam
import pytest

from silicone.multiple_infillers.split_collection_with_remainder_emissions import (
    SplitCollectionWithRemainderEmissions,
)
from silicone.utils import (
    _adjust_time_style_to_match,
    convert_units_to_MtCO2_equiv,
)

_msa = ["model_a", "scen_a"]
_msb = ["model_a", "scen_b"]


class TestSplitCollectionWithRemainderEmissions:
    tclass = SplitCollectionWithRemainderEmissions
    # tdb will generate test_db
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC", "kt C5F12-equiv/yr", 2.5, 4.5],
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
                "Emissions|HFC",
                "kt C5F12-equiv/yr",
                1.25,
                2,
                3,
            ],
            [
                "model_b",
                "scen_c",
                "World",
                "Emissions|HFC",
                "kt C5F12-equiv/yr",
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
            [
                "model_C",
                "scen_C",
                "World",
                "Emissions|KyotoTotal",
                "Mt CO2-equiv/yr",
                "",
                2.0,
                2,
                2,
            ],
            ["model_D", "scen_C", "World", "Emissions|CO2", "Mt CO2/yr", "", 2, 2, 2],
            [
                "model_D",
                "scen_C",
                "World",
                "Emissions|KyotoTotal",
                "Mt CO2-equiv/yr",
                "",
                2,
                2,
                2,
            ],
            ["model_D", "scen_F", "World", "Emissions|CO2", "Mt CO2/yr", "", 4, 4, 4],
            [
                "model_D",
                "scen_F",
                "World",
                "Emissions|KyotoTotal",
                "Mt CO2-equiv/yr",
                "",
                6,
                6,
                6,
            ],
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

    def test_infill_components_error_no_aggregate_data(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "The database to infill does not have the aggregate variable"
        )
        with pytest.raises(AssertionError, match=error_msg):
            tcruncher.infill_components("c", ["a"], "b", test_db)

    def test_db_error_no_info_aggregate(self, test_db):
        # test that crunching fails if there's no data about the remainder in the
        # database
        components = ["Emissions|HFC|C2F6"]
        aggregate = "Emissions|HFC"
        remainder = "Emissions|HFC|C5F12"
        infiller = self.tclass(test_db.filter(variable=aggregate, keep=False))

        error_msg = re.escape("No aggregate data in database.")
        with pytest.raises(AssertionError, match=error_msg):
            infiller.infill_components(
                aggregate, components, remainder, test_db.filter(variable=aggregate),
            )

    def test_db_error_existing_info(self, test_db):
        # test that crunching fails if there's already data about the remainder in the
        # database
        components = ["Emissions|HFC|C2F6"]
        aggregate = "Emissions|HFC"
        remainder = "Emissions|HFC|C5F12"
        infiller = self.tclass(test_db.filter(variable=components, keep=False))

        error_msg = re.escape(
            "The database to infill already has some component variables"
        )
        with pytest.raises(AssertionError, match=error_msg):
            infiller.infill_components(
                aggregate,
                components,
                remainder,
                test_db.filter(variable=components, keep=False),
            )

    def test_db_error_preexisting_follow_data(self, test_db):
        # test that crunching fails if there's no data about the follower gas in the
        # database
        components = ["Emissions|HFC|C2F6"]
        aggregate = "Emissions|HFC"
        remainder = "Emissions|HFC|C5F12"
        tcruncher = self.tclass(test_db)

        error_msg = re.escape(
            "The database to infill already has some component variables"
        )
        with pytest.raises(AssertionError, match=error_msg):
            tcruncher.infill_components(aggregate, components, remainder, test_db)

    def test_relationship_usage_works(self, test_db, test_downscale_df):
        # Test that we get the correct results when everything is in order.
        # First fix the units problem
        test_db = test_db.data
        test_db["unit"] = "kt C2F6-equiv/yr"
        test_db = pyam.IamDataFrame(test_db)
        test_downscale_df = test_downscale_df.data
        test_downscale_df["unit"] = "kt C2F6-equiv/yr"
        test_downscale_df = pyam.IamDataFrame(test_downscale_df)
        components = ["Emissions|HFC|C2F6"]
        aggregate = "Emissions|HFC"
        remainder = "Emissions|HFC|C5F12"
        infiller = self.tclass(test_db)
        # Fix times to agree
        test_downscale_df = _adjust_time_style_to_match(test_downscale_df, test_db)
        if test_db.time_col == "year":
            test_downscale_df.filter(
                year=test_db.data[test_db.time_col].values, inplace=True
            )
        else:
            test_downscale_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        # Perform the calculation
        filled = infiller.infill_components(
            aggregate, components, remainder, test_downscale_df
        )
        # The values returned should include only 1 entry per input entry, since there
        # is a single input component
        assert len(filled.data) == 8
        assert all([y in components + [remainder] for y in filled.variables()])
        assert np.allclose(
            filled.filter(variable=remainder)["value"].values
            + filled.filter(variable=components)["value"].values,
            test_downscale_df.filter(variable=aggregate)["value"].values,
        )

    def test_relationship_usage_works_multiple(self, test_db, test_downscale_df):
        # Test that the split emissions function works for slightly more complicated
        # data (two components).
        # Get matching times
        test_downscale_df = _adjust_time_style_to_match(test_downscale_df, test_db)
        if test_db.time_col == "year":
            test_downscale_df.filter(
                year=test_db.data[test_db.time_col].values, inplace=True
            )
        else:
            test_downscale_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        # Make the variables work for our case
        neg_component = "Emissions|HFC|CF4"
        pos_component = "Emissions|HFC|C2F6"
        add_to_test_db = test_db.filter(variable=pos_component)
        add_to_test_db["value"] = -add_to_test_db["value"]
        add_to_test_db["variable"] = neg_component
        test_db.append(add_to_test_db, inplace=True)
        components = ["Emissions|HFC|C2F6", neg_component]
        aggregate = "Emissions|HFC"
        remainder = "Emissions|HFC|C5F12"
        test_downscale_df.data["variable"] = aggregate
        test_db = convert_units_to_MtCO2_equiv(test_db)
        test_downscale_df = convert_units_to_MtCO2_equiv(test_downscale_df)
        infiller = self.tclass(test_db)
        filled = infiller.infill_components(
            aggregate, components, remainder, test_downscale_df
        )
        # The value returned should be a dataframe with 3 entries per original entry (4)
        assert len(filled.data) == 12
        assert all(y in filled.variables().values for y in components)
        assert all(y in components + [remainder] for y in filled.variables().values)
        # We also expect the amount of the variables to be conserved
        if test_db.time_col == "year":
            assert np.allclose(
                test_downscale_df.data.groupby("year").sum()["value"].values,
                filled.data.groupby("year").sum()["value"].values,
            )
        else:
            assert np.allclose(
                test_downscale_df.data.groupby("time").sum()["value"].values,
                filled.data.groupby("time").sum()["value"].values,
            )
        assert np.allclose(
            filled.filter(variable=neg_component)["value"].values,
            -filled.filter(variable=pos_component)["value"].values,
        )
        assert all(filled.filter(variable=neg_component)["value"].values < 0)

    def test_relationship_rejects_inconsistent_columns(self, larger_df, test_db):
        # There are optional extra columns on the DataFrame objects. This test ensures
        # that an error is thrown if we add together different sorts of DataFrame.
        aggregate = "Emissions|KyotoTotal"
        components = ["Emissions|CH4", "Emissions|N2O"]
        remainder = "Emissions|CO2"
        test_db = test_db.data
        test_db["variable"] = aggregate
        test_db = pyam.IamDataFrame(test_db)
        # larger_df has an extra column, "meta1"
        larger_df = _adjust_time_style_to_match(larger_df, test_db)
        infiller = self.tclass(larger_df)
        if test_db.time_col == "year":
            larger_df.filter(year=test_db.data[test_db.time_col].values, inplace=True)
        else:
            larger_df.filter(time=test_db.data[test_db.time_col], inplace=True)

        err_msg = re.escape(
            "The database and to_infill_db fed into this have inconsistent "
            "columns, which will prevent adding the data together properly."
        )
        with pytest.raises(AssertionError, match=err_msg):
            infiller.infill_components(aggregate, components, remainder, test_db)

    def test_relationship_works_with_unit_conversion(self, larger_df, test_db, caplog):
        # quiet pyam
        caplog.set_level(logging.ERROR, logger="pyam")

        # If we ask for N2O emissions when we don't have any in the input, we should
        # get a warning but also an output.
        aggregate = "Emissions|KyotoTotal"
        components = ["Emissions|CH4", "Emissions|N2O"]
        remainder = "Emissions|CO2"
        # Make the test data variables appropriate
        test_db = test_db.data
        test_db["variable"].iloc[0:2] = aggregate
        test_db["unit"] = "Mt CO2-equiv/yr"
        test_db = pyam.IamDataFrame(test_db)
        # We remove the extra column from the larger_df as it's not found in test_df
        larger_df = larger_df.data
        larger_df.drop("meta1", axis=1, inplace=True)
        larger_df = pyam.IamDataFrame(larger_df)
        larger_df = _adjust_time_style_to_match(larger_df, test_db)
        infiller = self.tclass(larger_df)
        if test_db.time_col == "year":
            larger_df.filter(year=test_db.data[test_db.time_col].values, inplace=True)
        else:
            larger_df.filter(time=test_db.data[test_db.time_col], inplace=True)
        with caplog.at_level(logging.INFO, logger="silicone.multiple_infillers"):
            returned = infiller.infill_components(
                aggregate, components, remainder, test_db
            )
        assert len(caplog.record_tuples) == 2
        assert len(returned.data) == len(test_db.filter(variable=aggregate).data) * 2
        # Make the data consistent:
        test_db = test_db.filter(variable="*Kyoto*")
        returned = infiller.infill_components(aggregate, components, remainder, test_db)
        assert len(returned.data) == 2 * len(test_db.filter(variable=aggregate).data)

        # Ensure that we get the same number if the unit conversion is done outside the
        # function
        infiller = self.tclass(
            convert_units_to_MtCO2_equiv(
                larger_df.filter(variable=[aggregate, remainder] + components)
            )
        )
        old_caplog = len(caplog.record_tuples)
        with caplog.at_level(logging.INFO, logger="silicone.multiple_infillers"):
            conv_returned = infiller.infill_components(
                aggregate, components, remainder, test_db
            )
        assert len(caplog.record_tuples) - old_caplog == 2
        assert all(conv_returned["unit"].unique() == "Mt CO2-equiv/yr")
        assert conv_returned.filter(variable=remainder).equals(
            returned.filter(variable=remainder)
        )
