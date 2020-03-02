from silicone.multiple_infillers.infill_composite_values import infill_composite_values
import numpy as np
import pandas as pd


class TestInfillCompositeValues:
    larger_df = pd.DataFrame(
        [
            [
                "model_C",
                "scen_C",
                "World",
                "Emissions|CO2|AFOLU",
                "Mt CO2/yr",
                "",
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
                "",
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
                "",
                2,
                2,
                2,
            ],
            ["model_D", "scen_C", "World", "Emissions|CH4", "Mt CH4/yr", "", 2, 2, 2],
            [
                "model_D",
                "scen_F",
                "World",
                "Emissions|CO2|Industry",
                "Mt CO2/yr",
                "",
                4,
                4,
                4,
            ],
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
            ["Emissions|PFC|*"]
        )

    def test_infill_composite_values_works(self, larger_df, caplog):
        # Ensure that the code performs correctly
        larger_df_copy = larger_df.copy()
        infilled = infill_composite_values(larger_df)
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
        # Ensure that the original is undisturbed by this operation
        assert larger_df_copy.equals(larger_df)
