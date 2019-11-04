import statistics

import pyam
import silicone.database_crunchers
import pandas as pd
import matplotlib.pyplot as plt

"""
This script measures how accurate the different crunchers are at recreating known data

"""
# ___________________________Input options____________________________________
# Where is the file stored for data used to fill in the sheet?
input_data = (
    "../Input/SSP_CMIP6_201811.csv"
)
# A list of all crunchers to investigate, here a reference to the actual cruncher
crunchers_list = [silicone.database_crunchers.DatabaseCruncherQuantileRollingWindows]
# This list must agree with the above list, but is the name of the crunchers
crunchers_name_list = ["QuantileRolling"]
# Leader is a single data class for the moment, but presented as a list.
leaders = ["CMIP6 Emissions|CO2"]
# This is the model/scenario combination to compare.
to_compare_filter = {'model': 'AIM/CGE', 'scenario': 'SSP3-70 (Baseline)'}
# Place to save the infilled data as a csvn
save_file = "../Output/CruncherComparison.csv"
# ____________________________end options____________________________________

assert len(crunchers_list) == len(crunchers_name_list)

db = pyam.IamDataFrame(input_data).filter(region="World")
real_results = db.filter(**to_compare_filter)
db.filter(**to_compare_filter, keep=False, inplace=True)
vars_to_crunch = [
    req for req in db.variables() if req not in leaders
]
# Initialise the object that holds the results
results_db = pd.DataFrame(index=vars_to_crunch, columns=crunchers_name_list)

for cruncher_ind in range(len(crunchers_list)):
    cruncher_instance = crunchers_list[cruncher_ind](db)
    for var_inst in vars_to_crunch:
        filler = cruncher_instance.derive_relationship(var_inst, leaders)
        interpolated = filler(real_results)
        originals = real_results.filter(variable=var_inst).data.set_index("year")["value"]
        # Currently I am normalising by the mean absolute value
        mean_abs = statistics.mean(abs(originals))
        interp_values = interpolated.data.set_index("year")["value"]
        assert originals.size == interp_values.size, \
            "Wrong number of values returned"
        assert interpolated['year'].size == interpolated['year'].unique().size, \
            "The wrong number of years have returned values"
        results_db[crunchers_name_list[cruncher_ind]][var_inst] = statistics.mean((interp_values-originals)**2)**0.5/mean_abs



