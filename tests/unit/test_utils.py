import re

import pandas as pd
import pytest

from silicone.utils import _get_unit_of_variable


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
