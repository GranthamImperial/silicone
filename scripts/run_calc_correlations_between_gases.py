import silicone.stats
from download_sr15_emissions import download_or_load_sr15


# This script shows how to use the investigatory script, _plot_emission_correlations_
# quantile_rolling_windows. It saves CSV files indicating statistics of correlation
# coefficients to a specified folder.
# ________________________________________________________
# We must indicate the main gas of interest
years = [2030, 2050, 2100]
output_dir = "../Output/Correlations"
# if non-null, also plot these quantiles.

# ________________________________________________________
# where do we get the data from?
SR15_SCENARIOS = "./sr15_scenarios_more_regions.csv"
sr15_data = download_or_load_sr15(SR15_SCENARIOS)
sr15_data = sr15_data.filter(region="World")

silicone.stats.calc_all_correlations(sr15_data, years, output_dir)
