import datetime
import re

import numpy as np
import pandas as pd
import pyam
import pytest

from silicone.database_crunchers.constant_ratio import ConstantRatio
from silicone.multiple_infillers.infill_all_required_emissions_for_openscm import (
    infill_all_required_variables,
)
from silicone.utils import _adjust_time_style_to_match


class TestGasDecomposeTimeDepRatio:
    _msa = ["model_a", "scen_a"]
    _msb = ["model_a", "scen_b"]
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 2, 3],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 0.5, 1.5],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
    )

    larger_df = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 2, 3],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 2],
            _msb + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 4, 5],
            _msb + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1.5, 2],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
    )

    def test_infillallrequiredvariables_works(self, test_db):
        output_times = list(set(test_db[test_db.time_col]))
        required_variables_list = ["Emissions|HFC|C5F12"]
        to_fill = test_db.filter(variable=required_variables_list, keep=False)
        output_df = infill_all_required_variables(
            to_fill,
            test_db,
            ["Emissions|HFC|C2F6"],
            required_variables_list,
            output_timesteps=output_times,
        )
        output_df.filter(variable=required_variables_list, keep=False).data.equals(
            to_fill.data
        )
        output_df.data.equals(test_db.data)

    def test_infillallrequiredvariables_doesnt_overwrite(self, test_db, larger_df):
        output_times = list(set(test_db[test_db.time_col]))
        correct_time_large_df = _adjust_time_style_to_match(larger_df, test_db)
        required_variables_list = ["Emissions|HFC|C5F12"]
        to_fill = test_db.copy()
        infilled = infill_all_required_variables(
            to_fill,
            correct_time_large_df,
            variable_leaders=["Emissions|HFC|C2F6"],
            output_timesteps=output_times,
            required_variables_list=required_variables_list,
        )
        assert infilled.data.equals(test_db.data)

    def test_infillallrequiredvariables_warning(self, test_db):
        output_times = list(set(test_db[test_db.time_col]))
        required_variables_list = ["Emissions|HFC|C5F12"]
        leader = ["Emissions|HFC|C2F6"]
        to_fill = test_db.filter(variable=required_variables_list, keep=False)
        database = test_db.copy()
        database.data["variable"].loc[
            database.data["variable"] == required_variables_list[0]
        ] = "Emissions|odd"
        err_msg = re.escape(
            "Missing some requested variables: {}".format(required_variables_list[0])
        )
        with pytest.warns(UserWarning):
            output_df = infill_all_required_variables(
                to_fill,
                database,
                variable_leaders=leader,
                output_timesteps=output_times,
                required_variables_list=required_variables_list,
            )
        assert all(output_df.filter(variable=leader, keep=False)["value"] == 0)
        # We should also get the same warning if we do not set an explicit
        # required_variables_list
        with pytest.warns(UserWarning):
            infill_all_required_variables(
                to_fill,
                database,
                variable_leaders=leader,
                output_timesteps=output_times,
            )

    def test_infillallrequiredvariables_changes_names(self, test_db):
        output_times = list(set(test_db[test_db.time_col]))
        required_variables_list = ["HFC|C5F12"]
        infilled_data_prefix = "Emissions"
        modified_test_db = test_db.copy()
        modified_test_db.data["variable"] = modified_test_db.data[
            "variable"
        ].str.replace(re.escape(infilled_data_prefix + "|"), "")
        to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
        output_df = infill_all_required_variables(
            to_fill,
            modified_test_db,
            ["HFC|C2F6"],
            required_variables_list,
            infilled_data_prefix=infilled_data_prefix,
            output_timesteps=output_times,
        )
        assert not output_df.filter(
            variable=required_variables_list, keep=False
        ).data.equals(to_fill.data)
        assert output_df.data.equals(test_db.data)

    def test_infillallrequiredvariables_check_results(self, test_db):
        required_variables_list = ["HFC|C5F12"]
        infilled_data_prefix = "Emissions"
        modified_test_db = test_db.copy()
        modified_test_db.data["variable"] = modified_test_db.data[
            "variable"
        ].str.replace(re.escape(infilled_data_prefix + "|"), "")
        to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
        output_times = to_fill[to_fill.time_col].unique()
        output_df = infill_all_required_variables(
            to_fill,
            modified_test_db,
            ["HFC|C2F6"],
            required_variables_list,
            infilled_data_prefix=infilled_data_prefix,
            output_timesteps=output_times,
            check_data_returned=True,
        )
        assert not output_df.filter(
            variable=required_variables_list, keep=False
        ).data.equals(to_fill.data)
        assert output_df.data.equals(test_db.data)

    def test_infillallrequiredvariables_check_results_use_old_prefix(self, test_db):
        required_variables_list = ["HFC|C5F12"]
        infilled_data_prefix = "Emissions"
        modified_test_db = test_db.copy()
        modified_test_db.data["variable"] = modified_test_db.data[
            "variable"
        ].str.replace(re.escape(infilled_data_prefix + "|"), "")
        to_fill = test_db.filter(variable=required_variables_list, keep=False)
        output_times = to_fill[to_fill.time_col].unique()
        output_df = infill_all_required_variables(
            to_fill,
            modified_test_db,
            ["HFC|C2F6"],
            required_variables_list,
            infilled_data_prefix=infilled_data_prefix,
            to_fill_old_prefix=infilled_data_prefix,
            output_timesteps=output_times,
            check_data_returned=True,
        )
        assert not output_df.filter(
            variable=required_variables_list, keep=False
        ).data.equals(to_fill.data)
        assert output_df.data.equals(test_db.data)

    def test_infillallrequiredvariables_check_results_fails_wrong_times(self, test_db):
        # We do not have data for all the default times, so this fails
        required_variables_list = ["HFC|C5F12"]
        infilled_data_prefix = "Emissions"
        modified_test_db = test_db.copy()
        modified_test_db.data["variable"] = modified_test_db.data[
            "variable"
        ].str.replace(re.escape(infilled_data_prefix + "|"), "")
        to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
        if test_db.time_col == "year":
            timesteps = None
        else:
            # No default times list is available.
            timesteps = [
                datetime.datetime(year=2010, day=15, month=6),
                datetime.datetime(year=2030, day=15, month=6),
            ]
        err_msg = re.escape("We do not have data for all required timesteps")
        with pytest.raises(AssertionError, match=err_msg):
            infill_all_required_variables(
                to_fill,
                modified_test_db,
                ["HFC|C2F6"],
                required_variables_list,
                infilled_data_prefix=infilled_data_prefix,
                check_data_returned=True,
                output_timesteps=timesteps,
            )

    def test_infillallrequiredvariables_check_results_error_bad_names_to_infill(
        self, test_db
    ):
        required_variables_list = ["HFC|C5F12"]
        infilled_data_prefix = "Emissions"
        modified_test_db = test_db.copy()
        modified_test_db.data["variable"] = modified_test_db.data[
            "variable"
        ].str.replace(re.escape(infilled_data_prefix + "|"), "")
        to_fill = modified_test_db.filter(
            variable=required_variables_list, keep=False
        ).append(test_db)
        err_msg = re.escape("Not all of the data begins with the expected prefix")
        with pytest.raises(ValueError, match=err_msg):
            output_df = infill_all_required_variables(
                to_fill,
                modified_test_db,
                ["HFC|C2F6"],
                required_variables_list,
                infilled_data_prefix=infilled_data_prefix,
                to_fill_old_prefix=infilled_data_prefix,
                check_data_returned=True,
                output_timesteps=[2010, 2015],
            )

    def test_infillallrequiredvariables_check_results_error_bad_names_in_database(
        self, test_db
    ):
        required_variables_list = ["HFC|C5F12"]
        infilled_data_prefix = "Emissions"
        modified_test_db = test_db.copy()
        modified_test_db.data["variable"] = modified_test_db.data[
            "variable"
        ].str.replace(re.escape(infilled_data_prefix + "|"), "")
        modified_test_db.append(test_db, inplace=True)
        to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
        if test_db.time_col == "year":
            output_times = [2010, 2015]
        else:
            output_times = [
                datetime.datetime(year=2010, day=15, month=6),
                datetime.datetime(year=2015, day=15, month=6),
            ]
        err_msg = re.escape(
            "This data already contains values with the expected final "
            "prefix. This suggests that some of it has already been infilled."
        )
        with pytest.raises(ValueError, match=err_msg):
            infill_all_required_variables(
                to_fill,
                modified_test_db,
                ["HFC|C2F6"],
                required_variables_list,
                infilled_data_prefix=infilled_data_prefix,
                check_data_returned=True,
                output_timesteps=output_times,
            )

    @pytest.mark.parametrize("additional_cols", [None, "Another_col"])
    def test_infillallrequiredvariables_check_results_interp_times(
        self, test_db, additional_cols
    ):
        # Check that we can get valid results at interpolated times
        required_variables_list = ["Emissions|HFC|C5F12"]
        leader = ["Emissions|HFC|C2F6"]
        if additional_cols:
            test_db.data[additional_cols] = 0
            test_db = pyam.IamDataFrame(test_db.data)
        if test_db.time_col == "year":
            output_times = [2012]
        else:
            # There is a leap year during 2015, so we subtract 3/5 of a day
            output_times = [pd.Timestamp(year=2012, day=14, month=6, hour=10)]
        to_fill = test_db.filter(variable=leader)
        output_df = infill_all_required_variables(
            to_fill,
            test_db,
            leader,
            required_variables_list,
            check_data_returned=True,
            output_timesteps=output_times,
        )
        # The values should be interpolations between the known values at the start
        assert np.isclose(
            output_df.data["value"][0], (3 * 0.5 + 2 * 1.5) / 5, atol=1e-5
        )
        assert np.isclose(output_df.data["value"][1], (3 * 2 + 2 * 3) / 5, atol=1e-5)

    def test_infillallrequiredvariables_check_results_kwargs(self, test_db):
        # Test that we can use the kwargs option for the multiple infiller
        required_variables_list = ["Emissions|HFC|C5F12"]
        leader = ["Emissions|HFC|C2F6"]
        kwargs = {"ratio": 1, "units": "Mt CO2-equiv/yr"}
        if test_db.time_col == "year":
            output_times = [2010]
        else:
            output_times = [datetime.datetime(year=2010, day=15, month=6)]
        to_fill = test_db.filter(variable=leader)
        output_df = infill_all_required_variables(
            to_fill,
            test_db,
            leader,
            required_variables_list,
            cruncher=ConstantRatio,
            check_data_returned=True,
            output_timesteps=output_times,
            **kwargs,
        )
        assert np.allclose(
            output_df.filter(variable=required_variables_list)["value"].values,
            output_df.filter(variable=leader)["value"].values,
        )
        assert (
            output_df.filter(variable=required_variables_list)["unit"].values
            == "Mt CO2-equiv/yr"
        )
