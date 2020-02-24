from silicone.multiple_infillers.infill_composite_values import infill_composite_values
import numpy as np
import pandas as pd


class TestInfillCompositeValues:
    larger_df = pd.DataFrame(
        [
            ["model_C", "scen_C", "World", "Emissions|CO2|AFOLU", "Mt CO2/yr", "", 1.5, 1.5, 1.5],
            ["model_C", "scen_C", "World", "Emissions|CO2|Industry", "Mt CO2/yr", "", 1, 1, 1],
            ["model_D", "scen_C", "World", "Emissions|CO2|Industry", "Mt CO2/yr", "", 2, 2, 2],
            ["model_D", "scen_C", "World", "Emissions|CH4", "Mt CH4/yr", "", 2, 2, 2],
            ["model_D", "scen_F", "World", "Emissions|CO2|Industry", "Mt CO2/yr", "", 4, 4, 4],
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

    def test_infill_composite_values_works(self, larger_df, caplog):
        with caplog.at_level("DEBUG"):
            infilled = infill_composite_values(larger_df)
        # Warnings are reported in
        assert caplog.record_tuples[0][2] == "No data found for {}".format(
            ["Emissions|PFC|*"]
        )
        assert np.allclose(
            infilled.filter(model="model_C", scenario="scen_C").data["value"],
            2.5
        )
        assert np.allclose(
            infilled.filter(
                model="model_D", scenario="scen_C", variable="Emissions|CO2"
            ).data["value"],
            2
        )
        assert np.allclose(
            infilled.filter(
                model="model_D", scenario="scen_C",
                variable="Emissions|Kyoto Gases (AR5-GWP100)"
            ).data["value"],
            2 + 2*28  # The 2*28 c
        )
