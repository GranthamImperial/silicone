import numpy as np
import pandas as pd
import pytest

from silicone.multiple_infillers.infill_composite_values import infill_composite_values
from silicone.utils import convert_units_to_MtCO2_equiv


class TestInfillCompositeValues:
    larger_df = pd.DataFrame(
        [
            [
                "model_C",
                "scen_C",
                "World",
                "Emissions|CO2|AFOLU",
                "Mt CO2/yr",
                1.5,
                1.5,
                1.5,
            ],
            [
                "model_C",
                "scen_C",
                "World",
                "Emissions|CO2|Industry",
                "Mt CO2/yr",
                1,
                1,
                1,
            ],
            [
                "model_D",
                "scen_C",
                "World",
                "Emissions|CO2|Industry",
                "Mt CO2/yr",
                2,
                2,
                2,
            ],
            ["model_D", "scen_C", "World", "Emissions|CH4", "Mt CH4/yr", 2, 2, 2],
            [
                "model_D",
                "scen_F",
                "World",
                "Emissions|CO2|Industry",
                "Mt CO2/yr",
                4,
                4,
                4,
            ],
            ["model_D", "scen_F", "World", "Emissions|CH4", "Mt CH4/yr", 2, 2, 2],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050,],
    )

    def test_infill_composite_values_warns(self, larger_df, caplog):
        with caplog.at_level("DEBUG"):
            infill_composite_values(
                larger_df, composite_dic={"Emissions|CO2": ["Emissions|CO2|*"]}
            )
        assert len(caplog.record_tuples) == 0
        with caplog.at_level("DEBUG"):
            infill_composite_values(larger_df)
        # Warnings are reported by the system for non-available data.
        assert caplog.record_tuples[0][2] == "No data found for {}".format(
            [
                "Emissions*|CF4",
                "Emissions*|C2F6",
                "Emissions*|C3F8",
                "Emissions*|C4F10",
                "Emissions*|C5F12",
                "Emissions*|C6F14",
                "Emissions*|C7F16",
                "Emissions*|C8F18",
            ]
        )

    def test_infill_composite_values_works(self, larger_df, caplog):
        # Ensure that the code performs correctly
        larger_df_copy = larger_df.copy()
        larger_df_copy.append(
            infill_composite_values(
                larger_df_copy, composite_dic={"Emissions|CO2": ["Emissions|CO2|*"]}
            ),
            inplace=True,
        )
        larger_df_copy = convert_units_to_MtCO2_equiv(larger_df_copy)
        infilled = infill_composite_values(larger_df_copy)
        assert np.allclose(
            infilled.filter(model="model_C", scenario="scen_C").data["value"], 2.5
        )
        assert np.allclose(
            infilled.filter(
                model="model_D", scenario="scen_C", variable="Emissions|CO2"
            ).data["value"],
            2,
        )
        assert np.allclose(
            infilled.filter(
                model="model_D",
                scenario="scen_F",
                variable="Emissions|Kyoto Gases (AR5-GWP100)",
            ).data["value"],
            4
            + 2 * 28,  # The 2*28 comes from the CH4, converted to CO2 equiv using AR5.
        )

    def test_infill_with_factors(self, larger_df):
        # Ensure that the multiplier works correctly
        larger_df_copy = larger_df.copy()
        half_industry_df = infill_composite_values(
            larger_df_copy,
            composite_dic={
                "Emissions|CO2": {
                    "Emissions|CO2|AFOLU": 1,
                    "Emissions|CO2|Industry": 0.5,
                }
            },
        )
        assert np.allclose(
            half_industry_df.filter(variable="Emissions|CO2")["value"].values,
            [2, 2, 2, 1, 1, 1, 2, 2, 2],
        )

    def test_infill_composite_values_subtraction(self, larger_df, caplog):
        # Ensure that the code performs correctly when we subtract emissions too
        larger_df_copy = larger_df.copy()
        larger_df_copy.append(
            infill_composite_values(
                larger_df_copy, composite_dic={"Emissions|CO2": ["Emissions|CO2|*"]}
            ),
            inplace=True,
        )
        AFOLU = "Emissions|CO2|AFOLU"
        larger_df_copy = convert_units_to_MtCO2_equiv(larger_df_copy)
        forgot_AFOLU = larger_df_copy.filter(variable=AFOLU, keep=False)
        infilled = infill_composite_values(
            forgot_AFOLU,
            composite_dic={AFOLU: {"Emissions|CO2": 1, "Emissions|CO2|Industry": -1}},
        )
        # We should have reconstructed the original data where it existed and also have
        # 0s now
        assert infilled.filter(model="model_C").data.equals(
            larger_df_copy.filter(variable=AFOLU).data.reset_index(drop=True)
        )
        assert np.allclose(infilled.filter(model="model_C", keep=False)["value"], 0)
