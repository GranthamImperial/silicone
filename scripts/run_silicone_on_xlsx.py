import pandas as pd
import pyam
import silicone.database_crunchers

"""
This script illustrates how to use silicone to augment the data fed into it by xlsx

"""
# ___________________________Input options____________________________________
# Where is the file stored for data used to fill in the sheet?
input_data_xlsx = \
    "../Input/MESSAGE-GLOBIOM_SSP1-19-SPA1-AR6_unharmonized Full-Med-Min sets(293).xlsx"
# Which sheet in the file should be read?
sheet_for_input = "Medium"
to_fill_xlsx = \
    "../Input/MESSAGE-GLOBIOM_SSP1-19-SPA1-AR6_unharmonized Full-Med-Min sets(293).xlsx"
# Which sheet in the file should be read?
sheet_to_fill = "Min"
# Here we specify the type of cruncher
type_of_cruncher = silicone.database_crunchers.DatabaseCruncherQuantileRollingWindows
# Leader is a single data class for the moment, but presented as a list.
leader = ["CEDS+|9+ Sectors|Emissions|CO2|Unharmonized"]

# Place to save the infilled data as a csv
save_file = "../Output/Infilling/infilled_data.csv"
# ____________________________end options____________________________________

df = pyam.IamDataFrame(input_data_xlsx, sheet_name=sheet_for_input)
to_fill = pyam.IamDataFrame(to_fill_xlsx, sheet_name=sheet_to_fill)

# Interpolate values in the input data if necessary
times_wanted = set(df[df.time_col])
# Currently the interpolate function seems broken? It isn't needed for our data set.
# for year in times_wanted:
#    df.interpolate(year)
#    to_fill.interpolate(year)

assert not df.data.isnull().any().any()
assert not to_fill.data.isnull().any().any()

required_variables = df.variables()
required_variables = [req for req in required_variables if req not in to_fill
    .variables().values]

cruncher = type_of_cruncher(df)
for req_var in required_variables:
    filler = cruncher.derive_relationship(req_var, leader)
    interpolated = filler(to_fill)
    # TODO: metadata joining is currently broken so this goes nowhere
    interpolated.set_meta(True, "interpolated")
    to_fill.append(interpolated)

# Some checks that we have fulfilled requirements
for _, (model, scenario) in to_fill[["model", "scenario"]].drop_duplicates().iterrows():
    msdf = to_fill.filter(model=model, scenario=scenario)
    msdf_variables = msdf["variable"].tolist()
    for v in required_variables:
        msvdf = msdf.filter(variable=v)
        msvdf_data = msvdf.data
        assert not msvdf_data.isnull().any().any()
        assert not msvdf_data.isempty
        assert all([y in msvdf_data["year"] for y in times_wanted])

to_fill.to_csv(save_file)


