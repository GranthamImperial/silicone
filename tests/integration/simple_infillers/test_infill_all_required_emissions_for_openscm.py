import re

import pytest
from silicone.simple_infillers.infill_all_required_emissions_for_openscm import InfillAllRequiredVariables
import pandas as pd
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

    def test_InfillAllRequiredVariables_works(self, test_db):
        # TODO: Ensure this works for datetimes too
        if test_db.time_col == "year":
            required_variables_list = ["Emissions|HFC|C5F12"]
            to_fill = test_db.filter(variable=required_variables_list, keep=False)
            output_df = InfillAllRequiredVariables(
                to_fill,
                test_db,
                ["Emissions|HFC|C2F6"],
                required_variables_list,
            )
            output_df.filter(variable=required_variables_list, keep=False).data.equals(
                to_fill.data
            )
            output_df.data.equals(
                test_db.data
            )

    def test_infill_all_fails_with_insufficient_input(self, test_db):
        if test_db.time_col == "year":
            required_variables_list = ["Emissions|HFC|C5F12"]
            to_fill = test_db.filter(variable=required_variables_list, keep=False)
            with pytest.raises(ValueError):
                InfillAllRequiredVariables(
                    to_fill,
                    test_db,
                    variable_leaders=["Emissions|HFC|C2F6"],
                    output_timesteps=[2010, 2015]
                )

    def test_infillallrequiredvariables_changes_names(self, test_db):
        # TODO: Ensure this works for datetimes too
        if test_db.time_col == "year":
            required_variables_list = ["HFC|C5F12"]
            infilled_data_prefix = "Emissions"
            modified_test_db = test_db.copy()
            modified_test_db.data["variable"] = modified_test_db.data["variable"].str.replace(
                re.escape(infilled_data_prefix + "|"), ""
            )
            to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
            output_df = InfillAllRequiredVariables(
                to_fill,
                modified_test_db,
                ["HFC|C2F6"],
                required_variables_list,
                infilled_data_prefix=infilled_data_prefix,
                output_timesteps=[2010, 2015],
            )
            assert not output_df.filter(variable=required_variables_list, keep=False).data.equals(
                to_fill.data
            )
            assert output_df.data.equals(
                test_db.data
            )

    def test_infillallrequiredvariables_check_results(self, test_db):
        # TODO: Ensure this works for datetimes too
        # This time we remove the prefix and only add it after the process has finished
        if test_db.time_col == "year":
            required_variables_list = ["HFC|C5F12"]
            infilled_data_prefix = "Emissions"
            modified_test_db = test_db.copy()
            modified_test_db.data["variable"] = modified_test_db.data["variable"].str.replace(
                re.escape(infilled_data_prefix + "|"), ""
            )
            to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
            output_df = InfillAllRequiredVariables(
                to_fill,
                modified_test_db,
                ["HFC|C2F6"],
                required_variables_list,
                infilled_data_prefix=infilled_data_prefix,
                output_timesteps=[2010, 2015],
                check_data_returned=True,
            )
            assert not output_df.filter(variable=required_variables_list, keep=False).data.equals(
                to_fill.data
            )
            assert output_df.data.equals(
                test_db.data
            )

    def test_infillallrequiredvariables_check_results_use_old_prefix(self, test_db):
        # TODO: Ensure this works for datetimes too
        if test_db.time_col == "year":
            required_variables_list = ["HFC|C5F12"]
            infilled_data_prefix = "Emissions"
            modified_test_db = test_db.copy()
            modified_test_db.data["variable"] = modified_test_db.data["variable"].str.replace(
                re.escape(infilled_data_prefix + "|"), ""
            )
            to_fill = test_db.filter(variable=required_variables_list, keep=False)
            output_df = InfillAllRequiredVariables(
                to_fill,
                modified_test_db,
                ["HFC|C2F6"],
                required_variables_list,
                infilled_data_prefix=infilled_data_prefix,
                to_fill_old_prefix=infilled_data_prefix,
                output_timesteps=[2010, 2015],
                check_data_returned=True,
            )
            assert not output_df.filter(variable=required_variables_list, keep=False).data.equals(
                to_fill.data
            )
            assert output_df.data.equals(
                test_db.data
            )


    def test_infillallrequiredvariables_check_results_fails_wrong_times(self, test_db):
        # We do not have data for all the default times, so this fails
        if test_db.time_col == "year":

            required_variables_list = ["HFC|C5F12"]
            infilled_data_prefix = "Emissions"
            modified_test_db = test_db.copy()
            modified_test_db.data["variable"] = modified_test_db.data["variable"].str.replace(
                re.escape(infilled_data_prefix + "|"), ""
            )
            to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
            err_msg = re.escape("We do not have data for all required timesteps")
            with pytest.raises(AssertionError, match=err_msg):
                InfillAllRequiredVariables(
                    to_fill,
                    modified_test_db,
                    ["HFC|C2F6"],
                    required_variables_list,
                    infilled_data_prefix=infilled_data_prefix,
                    check_data_returned=True,
                )


    def test_infillallrequiredvariables_check_results_error_bad_names_to_infill(
            self, test_db
    ):
        #
        if test_db.time_col == "year":

            required_variables_list = ["HFC|C5F12"]
            infilled_data_prefix = "Emissions"
            modified_test_db = test_db.copy()
            modified_test_db.data["variable"] = modified_test_db.data["variable"].str.replace(
                re.escape(infilled_data_prefix + "|"), ""
            )
            to_fill = modified_test_db.filter(variable=required_variables_list, keep=False).append(test_db)
            err_msg = re.escape(
                "Not all of the data begins with the expected prefix"
            )
            with pytest.raises(ValueError, match=err_msg):
                output_df = InfillAllRequiredVariables(
                    to_fill,
                    modified_test_db,
                    ["HFC|C2F6"],
                    required_variables_list,
                    infilled_data_prefix=infilled_data_prefix,
                    to_fill_old_prefix=infilled_data_prefix,
                    check_data_returned=True,
                    output_timesteps=[2010, 2015]
                )


    def test_infillallrequiredvariables_check_results_error_bad_names_in_database(
         self, test_db
    ):
        #
        if test_db.time_col == "year":

            required_variables_list = ["HFC|C5F12"]
            infilled_data_prefix = "Emissions"
            modified_test_db = test_db.copy()
            modified_test_db.data["variable"] = modified_test_db.data["variable"].str.replace(
                re.escape(infilled_data_prefix + "|"), ""
            )
            modified_test_db.append(test_db, inplace=True)
            to_fill = modified_test_db.filter(variable=required_variables_list, keep=False)
            err_msg = re.escape(
                "This data already contains values with the expected final "
                "prefix. This suggests that some of it has already been infilled."
            )
            with pytest.raises(ValueError, match=err_msg):
                output_df = InfillAllRequiredVariables(
                    to_fill,
                    modified_test_db,
                    ["HFC|C2F6"],
                    required_variables_list,
                    infilled_data_prefix=infilled_data_prefix,
                    check_data_returned=True,
                    output_timesteps=[2010, 2015],
                )