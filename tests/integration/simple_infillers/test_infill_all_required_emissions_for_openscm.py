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
                required_variables_list,
                variable_leaders=["Emissions|HFC|C2F6"],
                output_timesteps=[2010, 2015]
            )
            output_df.filter(variable=required_variables_list, keep=False).data.equals(
                to_fill.data
            )
            output_df.data.equals(
                test_db.data
            )



