import re
import os.path

import pyam
import numpy as np
import pandas as pd
import pytest

from silicone.utils import (
    _get_unit_of_variable,
    find_matching_scenarios,
    _make_interpolator,
    return_cases_which_consistently_split,
    convert_units_to_MtCO2_equiv,
    get_sr15_scenarios,
)

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


def test_return_cases_which_consistently_split_returns_nothing_with_no_data(
    check_aggregate_df
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
    # This time do not match the initial case, so we have to remove that to do the comparison.
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
    err_msg = "The units are unexpectedly not per year"
    with pytest.raises(AssertionError, match=err_msg):
        convert_units_to_MtCO2_equiv(limited_check_agg)


def test_convert_units_to_mtco2_equiv_fails_with_oom_units(check_aggregate_df):
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"].iloc[0] = "Tt CO2/yr"
    limited_check_agg = pyam.IamDataFrame(limited_check_agg.data)
    err_msg = "Unclear how to parse the units for {}.".format("Tt CO2")
    with pytest.raises(ValueError, match=err_msg):
        convert_units_to_MtCO2_equiv(limited_check_agg)


def test_convert_units_to_mtco2_equiv_fails_with_bad_units(check_aggregate_df):
    with pytest.raises(AssertionError):
        convert_units_to_MtCO2_equiv(check_aggregate_df)
    limited_check_agg = check_aggregate_df.filter(
        variable="Primary Energy*", keep=False
    )
    limited_check_agg.data["unit"].iloc[0] = "bad unit"
    with pytest.raises(AssertionError):
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


def test_get_files_and_use_them():
    SR15_SCENARIOS = "./sr15_scenarios_alternate.csv"
    valid_model_ids = [
        "MESSAGE*",
        "AIM*",
        "C-ROADS*",
        "GCAM*",
        # "IEA*",
        # "IMAGE*",
        # "MERGE*",
        # "POLES*",
        # "REMIND*",
        "WITCH*",
    ]
    if not os.path.isfile(SR15_SCENARIOS):
        get_sr15_scenarios(SR15_SCENARIOS, valid_model_ids)
    sr15_data = pyam.IamDataFrame(SR15_SCENARIOS)
    kyoto_list = ["Emissions|HFC", "Emissions|PFC"]
    target = "Emissions|F-Gases"
    valid_cases = return_cases_which_consistently_split(sr15_data, target, kyoto_list)
    len(valid_cases)
