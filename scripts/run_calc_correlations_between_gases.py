import silicone.stats
import pyam
import pandas as pd

# This script shows how to use the investigatory script, _plot_emission_correlations_
# quantile_rolling_windows. It saves CSV files indicating statistics of correlation
# coefficients to a specified folder.
# ________________________________________________________
# We must indicate the main gas of interest
years = range(2020, 2101, 10)
# Folder to save results:
output_dir = "../Output/Correlations"
# name of counts of variables
var_count_file = "variable_counts.csv"
# Location of data:
input_data = "./sr15_scenarios_more_regions.csv"
# If there is any prefix on the names of variables (before 'Emissions') it goes here
prefix = None  # "CMIP6 "
# We may need to run the following lines in order to download data:
"""
from download_sr15_emissions import download_or_load_sr15
SR15_SCENARIOS = "./sr15_scenarios_more_regions.csv"
download_or_load_sr15(SR15_SCENARIOS)
input_data = SR15_SCENARIOS
"""
# ________________________________________________________

sr15_data = pyam.IamDataFrame(input_data).filter(region="World")
if prefix:
    sr15_data.data["variable"] = sr15_data.data["variable"].apply(
        lambda x: x.replace(prefix, "")
    )

#This part counts the number of instances of a variable
maps = pd.Series(index=sr15_data.variables(False), dtype=int)
for var in sr15_data.variables(False):
    var_db = sr15_data.filter(variable=var)
    maps[var] = len(var_db.timeseries())
var_save_loc = output_dir + "/" + var_count_file
maps.to_csv(var_save_loc)
print("Counted the number of each variable and saved to ".format(var_save_loc))

# This part runs the correlations analysis
silicone.stats.calc_all_emissions_correlations(sr15_data, years, output_dir)
