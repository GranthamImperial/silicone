import silicone.stats
import pyam

# This script shows how to use the investigatory script, _plot_emission_correlations_
# quantile_rolling_windows. It saves CSV files indicating statistics of correlation
# coefficients to a specified folder.
# ________________________________________________________
# We must indicate the main gas of interest
years = range(2020, 2101, 10)
# Folder to save results:
output_dir = "../Output/Correlations"
# Location of data:
input_data = "./sr_15_complete.csv"
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

# This part runs the correlations analysis
silicone.stats.calc_all_emissions_correlations(sr15_data, years, output_dir)
