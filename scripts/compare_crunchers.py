import pyam
import silicone.database_crunchers as dc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

"""
This script measures how accurate the different crunchers are at recreating known data.
We remove the data of interest, infill to find it and compare the true and infilled 
values. 
"""
# __________________________________Input options_______________________________________
# Where is the file stored for data used to fill in the sheet?
input_data = "../Input/SSP_CMIP6_201811.csv"
# A list of all crunchers to investigate, here a reference to the actual cruncher
crunchers_list = [
    #  dc.DatabaseCruncherLeadGas,
    dc.DatabaseCruncherTimeDepRatio,
    dc.DatabaseCruncherQuantileRollingWindows,
    dc.DatabaseCruncherRMSClosest,
    dc.DatabaseCruncherLinearInterpolation,
]
options_list = [
    #  {},
    {"same_sign": True},
    {},
    {},
    {}
]
# This list must agree with the above list, but is the name of the crunchers
crunchers_name_list = [
    x.__name__.replace("DatabaseCruncher", "") for x in crunchers_list
]
# Leader is a single data class presented as a list.
leaders = ["CMIP6 Emissions|CO2"]
# Place to save the infilled data as a csv
save_file = "../Output/CruncherResults/CruncherComparisonLead_{}.csv".format(
    leaders[0].split("|")[-1]
)
# Do we want to save plots? If not, leave as None, else the location to save them.
# Note that these are not filter-dependent and so only the results of the last filter
# will persist
save_plots = None  #  "../Output/CruncherResults/plots/"
# Do we want to run this for all possible filters? If so, choose none,
# otherwise specify the filter here as a list of tuples
to_compare_filter = [
    ("GCAM4", "SSP4-34"),
    ("AIM/CGE", "SSP3-LowNTCF"),
]
# __________________________________end options_________________________________________

assert len(crunchers_list) == len(crunchers_name_list)
assert len(options_list) == len(crunchers_name_list)

db_all = pyam.IamDataFrame(input_data).filter(region="World")
# This is the model/scenario combination to compare.
if to_compare_filter:
    all_possible_filters = to_compare_filter
else:
    all_possible_filters = (
        db_all.data[["model", "scenario"]]
        .groupby(["model", "scenario"])
        .size()
        .index.values
    )
vars_to_crunch = [
    req for req in db_all.filter(level=1).variables() if req not in leaders
]
overall_results = pd.DataFrame(index=vars_to_crunch, columns=crunchers_name_list)
for one_filter in all_possible_filters:
    combo_filter = {"model": one_filter[0], "scenario": one_filter[1]}
    input_to_fill = db_all.filter(**combo_filter)
    # Remove all items that overlap directly with this
    db = db_all.filter(**combo_filter, keep=False)
    # Initialise the object that holds the results
    results_db = pd.DataFrame(index=vars_to_crunch, columns=crunchers_name_list)
    for cruncher_ind in range(len(crunchers_list)):
        cruncher_instance = crunchers_list[cruncher_ind](db)
        for var_inst in vars_to_crunch:
            var_units = db.filter(variable=var_inst).variables(True)["unit"]
            assert (
                var_units.size == 1
            ), "Multiple units involved, this spoils the calculation"
            filler = cruncher_instance.derive_relationship(
                var_inst, leaders, **options_list[cruncher_ind]
            )
            interpolated = filler(input_to_fill)
            originals = input_to_fill.filter(variable=var_inst).data.set_index("year")[
                "value"
            ]
            interp_values = interpolated.data.set_index("year")["value"]
            assert (
                originals.size == interp_values.size
            ), "Wrong number of values returned"
            assert (
                interpolated["year"].size == interpolated["year"].unique().size
            ), "The wrong number of years have returned values"
            # Set up normalisation
            norm_factor = pd.Series(index=interp_values.index, dtype=float)
            for year in norm_factor.index:
                norm_factor[year] = max(
                    db_all.filter(year=year, variable=var_inst).data["value"]
                ) - min(
                    db_all.filter(year=year, variable=var_inst).data["value"]
                )
            # Calculate the RMS difference, Normalised by the spread of values
            results_db[crunchers_name_list[cruncher_ind]][var_inst] = (
                np.nanmean(((interp_values - originals)[norm_factor > 0] / norm_factor[norm_factor > 0]) ** 2)
            ) ** 0.5
            if not np.isfinite(results_db[crunchers_name_list[cruncher_ind]][var_inst]):
                print("year: ".format(year))
            if save_plots:
                plt.close()
                ax = plt.subplot(111)
                plt.scatter(
                    x=originals.index, y=originals, label="True values", alpha=0.8
                )
                plt.scatter(
                    x=interp_values.index,
                    y=interp_values,
                    label="Model values",
                    marker="s",
                    alpha=0.5,
                )
                plt.xlabel("Year")
                plt.ylabel("Emissions of {} ({})".format(var_inst, var_units[0]))
                to_plot = db_all.filter(variable=var_inst)
                plt.scatter(
                    to_plot["year"],
                    to_plot["value"],
                    label="Other values",
                    alpha=0.5,
                    marker="v",
                    s=8,
                )
                plt.legend()
                plt.savefig(
                    save_plots
                    + "{}_{}.png".format(
                        crunchers_name_list[cruncher_ind], var_inst.split("|")[-1]
                    )
                )
    overall_results = (
        overall_results.fillna(0) + results_db.fillna(0) / len(all_possible_filters)
    )
overall_results.to_csv(save_file)
