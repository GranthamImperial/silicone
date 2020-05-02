import os
import re

import numpy as np
import pandas as pd
import pyam
import pytest
from openscm_units.unit_registry import ScmUnitRegistry
from pint.errors import UndefinedUnitError
from requests.exceptions import ConnectionError

from silicone.utils import (
    _construct_consistent_values,
    _get_unit_of_variable,
    _make_interpolator,
    convert_units_to_MtCO2_equiv,
    download_or_load_sr15,
    find_matching_scenarios,
    return_cases_which_consistently_split,
)

_ur = ScmUnitRegistry()
_ur.add_standards()

_mc = "model_c"
_sa = "scen_a"
_sb = "scen_b"
_eco2 = "Emissions|CO2"
_gtc = "Gt C/yr"
_ech4 = "Emissions|CH4"
_mtch4 = "Mt CH4/yr"
_msrvu = ["model", "scenario", "region", "variable", "unit"]
simple_df = pd.DataFrame(
    [
        [_mc, _sa, "World", _eco2, _gtc, 0, 1000, 5000],
        [_mc, _sb, "World", _eco2, _gtc, 1, 1000, 5000],
        [_mc, _sa, "World", _ech4, _mtch4, 0, 300, 500],
        [_mc, _sb, "World", _ech4, _mtch4, 1, 300, 500],
    ],
    columns=_msrvu + [2010, 2030, 2050],
)
simple_df = pyam.IamDataFrame(simple_df)
_msa = ["model_a", "scen_a"]
tdb = pd.DataFrame(
    [
        _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", 2, 3],
        _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 0.5, 1.5],
    ],
    columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
)
test_db = pyam.IamDataFrame(tdb)

df_low = simple_df.copy()
df_low.data["scenario"].loc[df_low.data["scenario"] == "scen_a"] = "right_scenario"
df_low.data["scenario"].loc[df_low.data["scenario"] == "scen_b"] = "wrong_scenario"
df_high = df_low.copy()
df_high["model"] = "high_model"
df_low.data["value"] = df_low.data["value"] - 10
df_high.data["value"] = df_high.data["value"] + 11
df_near = simple_df.copy()
df_near.data["value"] = df_near["value"] + 1
df_to_test = df_low.append(df_near).append(df_high)
# We need to refresh the metadata in order to proceed.
df_to_test = pyam.IamDataFrame(df_to_test.data)

variable_leaders = ["Emissions|CO2"]
variable_follower = "Emissions|CH4"


@pytest.mark.parametrize(
    "half_val, expected", [(0.5, "scen_a"), (0.49, "scen_a"), (0.51, "scen_b")]
)
def test_find_matching_scenarios_matched(half_val, expected):
    # Tests
    # 1) that 1st option is used in the case of equality
    # 2) and if it's closer,
    # 3) But not if it's further away
    half_simple_df = simple_df.filter(scenario="scen_a")
    half_simple_df.data["value"].loc[0] = half_val
    scenarios = find_matching_scenarios(
        simple_df,
        half_simple_df,
        variable_follower,
        variable_leaders,
        ["scen_a", "scen_b"],
    )
    assert scenarios == ("*", expected)


def test_find_matching_scenarios_no_data_for_time():
    time_col = simple_df.time_col
    half_simple_df = simple_df.filter(scenario="scen_a")
    half_simple_df.data[time_col].loc[0] = 0
    with pytest.raises(ValueError):
        find_matching_scenarios(
            simple_df,
            half_simple_df,
            variable_follower,
            variable_leaders,
            ["scen_a", "scen_b"],
        )


def test_find_matching_scenarios_use_change_instead_of_absolute():
    # In this case, we will ignore any offset
    half_simple_df = simple_df.filter(scenario="scen_a")
    half_simple_df.data["value"] = half_simple_df.data["value"] + 10000
    scenarios = find_matching_scenarios(
        simple_df,
        half_simple_df,
        variable_follower,
        variable_leaders,
        ["scen_a", "scen_b"],
        use_change_not_abs=True,
    )
    assert scenarios == ("*", "scen_a")


@pytest.mark.parametrize(
    "options,expected",
    [
        (["scen_a", "scen_b", "right_scenario"], "right_scenario"),
        (
            ["non-existant", "wrong_scenario", "scen_a", "scen_b", "right_scenario"],
            "wrong_scenario",
        ),
        (["right_scenario", "wrong_scenario", "scen_a", "scen_b"], "right_scenario"),
    ],
)
def test_find_matching_scenarios_complicated(options, expected):
    # This is similar to the above case except with multiple models involved and
    # requiring specific interpolation. Tests:
    # 1) Closest option chosen
    # 2) Invalid options ignored, if tied the earlier option is selected instead
    # 3) This reverses as expected
    scenario = find_matching_scenarios(
        df_to_test, simple_df, variable_follower, variable_leaders, options
    )
    assert scenario == ("*", expected)


def test_find_matching_scenarios_dual_region():
    multiregion_df = simple_df.data.append(
        pd.DataFrame(
            [[_mc, _sa, "Country", _eco2, _gtc, 2010, 2]],
            columns=_msrvu + [simple_df.time_col, "value"],
        )
    )
    multiregion_df = pyam.IamDataFrame(multiregion_df)
    with pytest.raises(AssertionError):
        find_matching_scenarios(
            df_to_test,
            multiregion_df,
            variable_follower,
            variable_leaders,
            ["right_scenario", "wrong_scenario", "scen_a", "scen_b"],
            return_all_info=True,
        )


def test_find_matching_scenarios_empty():
    noregion_df = simple_df.filter(scenario="impossible")
    nothing = find_matching_scenarios(
        df_to_test,
        noregion_df,
        variable_follower,
        variable_leaders,
        ["right_scenario", "wrong_scenario", "scen_a", "scen_b"],
        return_all_info=True,
    )
    assert nothing is None


def test_find_matching_scenarios_get_precise_values():
    # We should get out explicit numbers if we ask for them
    all_data = find_matching_scenarios(
        df_to_test,
        simple_df,
        variable_follower,
        variable_leaders,
        ["right_scenario", "wrong_scenario", "scen_a", "scen_b"],
        return_all_info=True,
    )
    assert all_data[0][1] == all_data[1][1]
    assert all_data[0][1] == 0
    assert all_data[2][1] > 0
    unsplit_model_c = all_data[2][1]
    # And if we separate out by model, we no longer combine the high and low values
    # to produce a good match
    all_data = find_matching_scenarios(
        df_to_test,
        simple_df,
        variable_follower,
        variable_leaders,
        ["right_scenario", "wrong_scenario", "scen_a", "scen_b"],
        classify_models=["high_model", _mc],
        return_all_info=True,
    )
    assert all_data[-1][0][0] == "high_model"
    assert all_data[0][0][0] == _mc
    assert all_data[0][1] == unsplit_model_c


def test_find_matching_scenarios_differential():
    # If we use a differential measurement, they should all be the same in this case
    all_data = find_matching_scenarios(
        df_to_test,
        simple_df,
        variable_follower,
        variable_leaders,
        ["right_scenario", "wrong_scenario", "scen_a", "scen_b"],
        classify_models=["high_model", _mc],
        return_all_info=True,
        use_change_not_abs=True,
    )
    assert all_data[0][0] == ("high_model", "right_scenario")
    assert all_data[0][1] == all_data[5][1]
    # But if we add a small amount to only one point in the differential, it will
    # be downgraded
    df_to_test["value"].iloc[0] = df_to_test["value"].iloc[0] + 0.1
    all_data = find_matching_scenarios(
        df_to_test,
        simple_df,
        variable_follower,
        variable_leaders,
        ["right_scenario", "wrong_scenario", "scen_a", "scen_b"],
        classify_models=["high_model", _mc],
        return_all_info=True,
        use_change_not_abs=True,
    )
    assert all_data[0][0] == ("high_model", "right_scenario")
    assert all_data[0][1] != all_data[1][1]
    df_to_test["value"].iloc[0] = df_to_test["value"].iloc[0] - 0.6
    all_data = find_matching_scenarios(
        df_to_test,
        simple_df,
        variable_follower,
        variable_leaders,
        ["right_scenario", "wrong_scenario", "scen_a", "scen_b"],
        classify_models=["high_model", _mc],
        return_all_info=True,
        use_change_not_abs=True,
    )
    assert all_data[0][0] == (_mc, "right_scenario")
    assert all_data[0][1] == all_data[1][1]


def test__make_interpolator():
    variable_leaders = "variable_leaders"
    variable_follower = "variable_follower"
    time_col = "years"
    x_set = np.array([1, 1, 2, 3])
    y_set = np.array([6, 4, 3, 2])
    times = np.array([1, 1, 1, 1])
    wide_db = pd.DataFrame(
        {variable_leaders: x_set, variable_follower: y_set, time_col: times}
    )

    # Illustrate the expected relationship between the numbers above, mapping 1 to
    # the average of 6 and 4, i.e. 5.
    input = np.array([5, 4, 3, 2, 2.5, 1, 0])
    expected_output = np.array([2, 2, 2, 3, 2.5, 5, 5])
    interpolator = _make_interpolator(
        variable_follower, variable_leaders, wide_db, time_col
    )
    output = interpolator[1](input)
    np.testing.assert_allclose(output, expected_output, atol=1e-10)


@pytest.mark.parametrize(
    "var,exp",
    (
        ["Emissions|CO2", "Mt CO2/yr"],
        ["Primary Energy", "EJ/y"],
        ["Primary Energy|*", "EJ/y"],
    ),
)
def test_get_unit_of_variable(var, exp, check_aggregate_df):
    assert _get_unit_of_variable(check_aggregate_df, var) == [exp]


def test_get_unit_of_variable_error(check_aggregate_df):
    edf = check_aggregate_df.filter(variable="Emissions|CH4").timeseries()
    edf *= 0.5
    edf = edf.reset_index()
    edf["unit"] = "Mt C/yr"
    edf["model"] = "tweaked unit"

    check_aggregate_df = check_aggregate_df.append(edf)

    error_msg = re.escape("`Emissions|CH4` has multiple units")
    with pytest.raises(AssertionError, match=error_msg):
        _get_unit_of_variable(check_aggregate_df, "Emissions|CH4")

    assert sorted(
        _get_unit_of_variable(
            check_aggregate_df, "Emissions|CH4", multiple_units="continue"
        )
    ) == sorted(["Mt CH4/yr", "Mt C/yr"])


def test_return_cases_which_consistently_split_works(check_aggregate_df):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    cases = return_cases_which_consistently_split(limited_check_agg, "*CO2", ["*CO2*"])
    assert pd.DataFrame(cases, columns=["model", "scenario", "region"]).equals(
        limited_check_agg.data[["model", "scenario", "region"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )


def test_return_cases_which_consistently_split_bad_to_split(check_aggregate_df):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    cases = return_cases_which_consistently_split(limited_check_agg, "Junk", ["*CO2*"])
    assert len(cases) == 0


def test_return_cases_which_consistently_split_numerical_error(check_aggregate_df):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["value"] = limited_check_agg.data[
        "value"
    ] + np.random.normal(0, 0.0001, len(limited_check_agg.data["value"]))
    cases = return_cases_which_consistently_split(limited_check_agg, "*CO2", ["*CO2*"])
    assert pd.DataFrame(cases, columns=["model", "scenario", "region"]).equals(
        limited_check_agg.data[["model", "scenario", "region"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    how_close = {"equal_nan": True, "rtol": 1e-16}
    cases = return_cases_which_consistently_split(
        limited_check_agg, "*CO2", ["*CO2*"], how_close=how_close
    )
    assert not cases


def test_return_cases_which_consistently_split_returns_nothing_with_no_data(
    check_aggregate_df,
):
    cases = return_cases_which_consistently_split(
        check_aggregate_df, "not_here", ["also_not_here"]
    )
    assert not cases


def test_return_cases_which_consistently_split_one_fails(check_aggregate_df):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["value"].iloc[0] = 41
    cases = return_cases_which_consistently_split(limited_check_agg, "*CO2", ["*CO2*"])
    # This time do not match the initial case, so we have to remove that to do the
    # comparison.
    assert pd.DataFrame(cases, columns=["model", "scenario", "region"]).equals(
        limited_check_agg.data[["model", "scenario", "region"]]
        .drop_duplicates()
        .iloc[1:]
        .reset_index(drop=True)
    )


def test_convert_units_to_mtco2_equiv_fails_with_month_units(check_aggregate_df):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"].iloc[0] = "Mt CH4/mo"
    limited_check_agg = pyam.IamDataFrame(limited_check_agg.data)
    err_msg = "'mo' is not defined in the unit registry"
    with pytest.raises(UndefinedUnitError, match=err_msg):
        convert_units_to_MtCO2_equiv(limited_check_agg)


def test_convert_units_to_mtco2_equiv_fails_with_oom_units(check_aggregate_df):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"].iloc[0] = "Tt CO2"
    limited_check_agg = pyam.IamDataFrame(limited_check_agg.data)
    err_msg = re.escape(
        "Cannot convert from Tt CO2 (cleaned is: Tt CO2) to Mt CO2-equiv/yr (cleaned is: Mt CO2/yr)"
    )
    with pytest.raises(ValueError, match=err_msg):
        convert_units_to_MtCO2_equiv(limited_check_agg)


def test_convert_units_to_mtco2_equiv_fails_with_bad_units(check_aggregate_df):
    err_msg = "'y' is not defined in the unit registry"
    with pytest.raises(UndefinedUnitError, match=err_msg):
        convert_units_to_MtCO2_equiv(check_aggregate_df)

    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"].iloc[0] = "bad unit"
    err_msg = "'bad' is not defined in the unit registry"
    with pytest.raises(UndefinedUnitError, match=err_msg):
        convert_units_to_MtCO2_equiv(limited_check_agg)


@pytest.mark.parametrize(
    "ARoption,expected", [(False, [28, 6.63]), (True, [25, 7.390])]
)
def test_convert_units_to_MtCO2_equiv_works(check_aggregate_df, ARoption, expected):
    # ARoption turns the use of AR4 on, rather than AR5 (the default)
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    converted_units = convert_units_to_MtCO2_equiv(limited_check_agg, ARoption)
    assert all(y[:6] == "Mt CO2" for y in converted_units.data["unit"].unique())
    # Index 1 is already in CO2
    assert (
        converted_units.data["value"].loc[1] == limited_check_agg.data["value"].loc[1]
    )
    # At index 122 we are in units of Mt methane, rate 28* higher in AR5
    assert np.isclose(
        converted_units.data["value"].loc[122],
        limited_check_agg.data["value"].loc[122] * expected[0],
    )
    # At index 142 we have kt CF4, 6630 times more effective/kg but / 1000 for k -> G
    assert np.isclose(
        converted_units.data["value"].loc[142],
        limited_check_agg.data["value"].loc[142] * expected[1],
    )


@pytest.mark.parametrize(
    "unit_start", ("kt CF4-equiv/yr", "kt CF4 equiv/yr", "kt CF4equiv/yr",)
)
def test_convert_units_to_MtCO2_equiv_equiv_start(check_aggregate_df, unit_start):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"] = unit_start
    converted_data = convert_units_to_MtCO2_equiv(limited_check_agg)

    assert (converted_data.data["unit"] == "Mt CO2-equiv/yr").all()

    with _ur.context("AR5GWP100"):
        exp_conv_factor = _ur("kt CF4/yr").to("Mt CO2/yr").magnitude
    assert converted_data.data["value"].equals(
        limited_check_agg.data["value"] * exp_conv_factor
    )


def test_convert_units_to_MtCO2_equiv_doesnt_change(check_aggregate_df):
    # Check that it does nothing when nothing needs doing
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"] = "Mt CO2-equiv/yr"
    converted_data = convert_units_to_MtCO2_equiv(limited_check_agg)
    assert (converted_data.data["unit"] == "Mt CO2-equiv/yr").all()
    assert converted_data.data.equals(limited_check_agg.data)


def test_convert_units_to_MtCO2_doesnt_change(check_aggregate_df):
    # Check that it does nothing when nothing needs doing
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"] = "Mt CO2/yr"
    converted_data = convert_units_to_MtCO2_equiv(limited_check_agg)

    assert (converted_data.data["unit"] == "Mt CO2/yr").all()
    assert converted_data.data.equals(limited_check_agg.data)


def test_get_files_and_use_them():
    try:
        # Remove any pre-exising files to check we make new ones
        SR15_SCENARIOS = "./sr15_scenarios.csv"
        if os.path.isfile(SR15_SCENARIOS):
            os.remove(SR15_SCENARIOS)
        valid_model_ids = ["GCAM*"]
        sr15_data = download_or_load_sr15(SR15_SCENARIOS, valid_model_ids)
        min_expected_var = [
            "Emissions|N2O",
            "Emissions|CO2",
            "Emissions|CH4",
            "Emissions|F-Gases",
        ]
        variables_in_result = sr15_data.variables()
        assert all([y in variables_in_result.values for y in min_expected_var])
        assert os.path.isfile(SR15_SCENARIOS)
        # Now check that the function works correctly again
        variables_in_result_2 = download_or_load_sr15(
            SR15_SCENARIOS, valid_model_ids
        ).variables()
        assert all(variables_in_result_2 == variables_in_result)
        blank_variables = download_or_load_sr15(
            SR15_SCENARIOS, ["bad_model"]
        ).variables()
        assert all([y not in blank_variables.values for y in min_expected_var])
        os.remove(SR15_SCENARIOS)
    except ConnectionError as e:
        pytest.skip("Could not connect to the IIASA database: {}".format(e))


def test__construct_consistent_values():
    test_db_co2 = convert_units_to_MtCO2_equiv(test_db)
    aggregate_name = "agg"
    assert aggregate_name not in test_db_co2.variables().values
    component_ratio = ["Emissions|HFC|C2F6", "Emissions|HFC|C5F12"]
    consistent_vals = _construct_consistent_values(
        aggregate_name, component_ratio, test_db_co2
    )
    assert aggregate_name in consistent_vals["variable"].values
    consistent_vals = consistent_vals.timeseries()
    timeseries_data = test_db_co2.timeseries()
    assert all(
        [
            np.allclose(
                consistent_vals.iloc[0].iloc[ind],
                timeseries_data.iloc[0].iloc[ind] + timeseries_data.iloc[1].iloc[ind],
            )
            for ind in range(len(timeseries_data.iloc[0]))
        ]
    )


def test__construct_consistent_values_with_equiv():
    test_db_co2 = convert_units_to_MtCO2_equiv(test_db)
    test_db_co2.data["unit"].loc[0:1] = "Mt CO2/yr"
    aggregate_name = "agg"
    assert aggregate_name not in test_db_co2.variables().values
    component_ratio = ["Emissions|HFC|C2F6", "Emissions|HFC|C5F12"]
    consistent_vals = _construct_consistent_values(
        aggregate_name, component_ratio, test_db_co2
    )
    assert aggregate_name in consistent_vals["variable"].values
    consistent_vals = consistent_vals.timeseries()
    timeseries_data = test_db_co2.timeseries()
    assert all(
        [
            np.allclose(
                consistent_vals.iloc[0].iloc[ind],
                timeseries_data.iloc[0].iloc[ind] + timeseries_data.iloc[1].iloc[ind],
            )
            for ind in range(len(timeseries_data.iloc[0]))
        ]
    )
    # We also require that the output units are '-equiv'
    assert all(
        y == "Mt CO2-equiv/yr" for y in consistent_vals.index.get_level_values("unit")
    )


def test_construct_consistent_error_multiple_units():
    # test that construction fails if there's no data about the follower gas in the
    # database
    aggregate_name = "Emissions|HFC|C5F12"
    components = ["Emissions|HFC|C2F6"]
    test_db_units = test_db.copy()
    test_db_units.data["variable"] = components[0]
    error_msg = re.escape(
        "Too many units found to make a consistent {}".format(aggregate_name)
    )
    with pytest.raises(ValueError, match=error_msg):
        _construct_consistent_values(aggregate_name, components, test_db_units)


def test_construct_consistent_error_no_data():
    # test that construction fails if there's no data about the follower gas in the
    # database
    aggregate_name = "Emissions|HFC|C5F12"
    components = ["Emissions|HFC|C2F6"]
    test_db_ag = test_db.filter(variable="not there")  # This generates an empty df
    error_msg = re.escape(
        "Attempting to construct a consistent {} but none of the components "
        "present".format(aggregate_name)
    )
    with pytest.raises(ValueError, match=error_msg):
        _construct_consistent_values(aggregate_name, components, test_db_ag)
