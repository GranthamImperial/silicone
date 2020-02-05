import logging
import re

import pyam
import tqdm
from silicone.database_crunchers import DatabaseCruncherQuantileRollingWindows, DatabaseCruncherConstantRatio

def InfillAllRequiredVariables(
        to_fill,
        database,
        variable_leaders,
        required_variables_list=None,
        cruncher=DatabaseCruncherQuantileRollingWindows,
        output_timesteps=None,
        infilled_data_prefix=None,
        to_fill_old_prefix=None,
        check_data_returned=False,
):
    # Use default arguments for unfilled options:
    if output_timesteps is None:
        output_timesteps = [2015] + list(range(2020, 2101, 10))
    if required_variables_list is None:
        required_variables_list = [
            "Emissions|HFC|HFC245ca"
            "Emissions|BC",
            "Emissions|HFC|HFC125",
            "Emissions|PFC|CF4",
            "Emissions|PFC|C2F6",
            "Emissions|PFC|C6F14",
            "Emissions|CH4",
            "Emissions|CO2|AFOLU",
            "Emissions|CO2|Energy and Industrial Processes",
            "Emissions|CO",
            "Emissions|HFC|HFC134a",
            "Emissions|HFC|HFC143a",
            "Emissions|HFC|HFC227ea",
            "Emissions|HFC|HFC23",
            "Emissions|HFC|HFC32",
            "Emissions|HFC|HFC43-10",
            "Emissions|N2O",
            "Emissions|NH3",
            "Emissions|NOx",
            "Emissions|OC",
            "Emissions|SF6",
            "Emissions|Sulfur",
            "Emissions|VOC",
        ]

    # Check that the input is valid
    if to_fill_old_prefix:
        if any(
            to_fill.data["variable"].map(lambda x: x[: len(to_fill_old_prefix)])
            != to_fill_old_prefix
        ):
            raise ValueError("Not all of the data begins with the expected prefix")
        to_fill.data["variable"] = to_fill.data["variable"].str.replace(
            re.escape(to_fill_old_prefix + "|"), ""
        )
    if infilled_data_prefix:
        if any(
            to_fill.data["variable"].map(lambda x: x[: len(infilled_data_prefix)])
            == infilled_data_prefix
        ):
            raise ValueError(
                "This data already contains values with the expected final "
                "prefix. This suggests that some of it has already been infilled."
            )
    assert len(to_fill.regions()) == 1
    assert len(database.regions()) == 1
    assert (
            to_fill.data["region"][0] == database.data["region"][0]
    ), "The cruncher data and the infilled data have different regions."
    # Perform any interpolations required here
    timecol = database.time_col
    assert timecol == to_fill.time_col
    df_times_missing = database.timeseries().isna().sum() > 0
    to_fill_times_missing = to_fill.timeseries().isna().sum() > 0
    for time in output_timesteps:
        if time not in database[timecol].tolist() or df_times_missing[time]:
            # TODO: ensure that this works with date-times too.
            database.interpolate(time)
        if time not in to_fill[timecol].tolist() or to_fill_times_missing[time]:
            to_fill.interpolate(time)
    # Filter for desired times
    if timecol == "year":
        database = database.filter(year=output_timesteps)
        to_fill = to_fill.filter(year=output_timesteps)
    else:
        database = database.filter(time=output_timesteps)
        to_fill = to_fill.filter(time=output_timesteps)
    # Infill unavailable data
    assert not database.data.isnull().any().any()
    assert not to_fill.data.isnull().any().any()
    to_fill_orig = to_fill.copy()
    unavailable_variables = [variab for variab in required_variables_list if variab not in database.variables().values]
    if unavailable_variables:
        Warning("No data for {}".format(unavailable_variables))
        # Infill the required variables with 0s.
        kwarg_dict = {"ratio": 0, "units": "Mt Co2-equiv/yr"}
        to_fill = _perform_crunch_and_check(
            unavailable_variables,
            variable_leaders,
            to_fill,
            database,
            DatabaseCruncherConstantRatio,
            output_timesteps,
            to_fill_orig,
            check_data_returned=False,
            **kwarg_dict
        )
    available_variables = [variab for variab in required_variables_list if variab in database.variables().values]
    if available_variables:
        to_fill = _perform_crunch_and_check(
            available_variables,
            variable_leaders,
            to_fill,
            database,
            cruncher,
            output_timesteps,
            to_fill_orig,
            check_data_returned=check_data_returned
        )
    if infilled_data_prefix:
        to_fill.data["variable"] = infilled_data_prefix + "|" + to_fill.data["variable"]
    return to_fill


def _perform_crunch_and_check(
        required_variables,
        leaders,
        to_fill,
        df,
        type_of_cruncher,
        output_timesteps,
        to_fill_orig,
        check_data_returned=False,
        **kwargs
):
    """
        Takes a list of scenarios to infill and infills them according to the options
        presented.

        Parameters
        ----------
        required_variables : List(str)
            The variable names to infill
        leaders : List(str)
            The leaders to guide the infilling
        to_fill : IamDataFrame
            The data frame to infill
        df : IamDataFrame
            The data frame to base the infilling on
        type_of_cruncher : :obj: silicone cruncher
            the silicone package cruncher class to use for the infilling
        output_timesteps : list(int or datetime)
            When there should be data returned. Time-based interpolation will occur if
            this is more frequent than the data allows, data will be filtered out if
            there is additional time information.
        to_fill_orig : IamDataFrame
            The original, unfiltered and unaltered data input. We use this for
            performing checks.
        kwargs : Dict
            Any key word arguments to include in the cruncher calculation

        Returns
        -------
        :obj:IamDataFrame
            The infilled dataframe
        """
    if not all(x in df.variables().values for x in required_variables):
        not_present = [x for x in required_variables if
                       x not in df.variables().values]
        raise ValueError(
            "Missing some requested variables: {}".format(not_present))
    cruncher = type_of_cruncher(df)
    for req_var in tqdm.tqdm(required_variables,
                             desc="Filling required variables"):
        interpolated = _infill_variable(cruncher, req_var, leaders, to_fill, **kwargs)
        if interpolated:
            to_fill = to_fill.append(interpolated)
    # Optionally check we have added all the required data
    if not check_data_returned:
        return to_fill
    for _, (model, scenario) in (
            to_fill[["model", "scenario"]].drop_duplicates().iterrows()
    ):
        msdf = to_fill.filter(model=model, scenario=scenario)
        for v in required_variables:
            msvdf = msdf.filter(variable=v)
            msvdf_data = msvdf.data
            assert not msvdf_data.isnull().any().any()
            assert not msvdf_data.empty
            assert all(
                [y in msvdf_data[df.time_col].values for y in output_timesteps]
            ), "We do not have data for all required timesteps"

    # Check no data was overwritten by accident
    for model in tqdm.tqdm(
            to_fill_orig.models(),
            desc="Consistency with original model data checks"
    ):
        mdf = to_fill_orig.filter(model=model,
                                  variable=leaders + required_variables)
        for scenario in mdf.scenarios():
            msdf = mdf.filter(scenario=scenario)
            msdf_filled = to_fill.filter(
                model=model, scenario=scenario,
                variable=msdf["variable"].unique()
            )

            common_times = set(msdf_filled[msdf.time_col]).intersection(msdf[msdf.time_col])
            if common_times:
                if msdf.time_col == "year":
                    msdf = msdf.filter(year=list(common_times))
                    msdf_filled = msdf_filled.filter(year=list(common_times))
                else:
                    msdf = msdf.filter(time=list(common_times))
                    msdf_filled = msdf_filled.filter(time=list(common_times))
                assert pyam.compare(msdf, msdf_filled).empty
    return to_fill


def _infill_variable(cruncher_i, req_variable, leader_i, to_fill_i, **kwargs):
    filler = cruncher_i.derive_relationship(req_variable, leader_i, **kwargs)
    # only fill for scenarios who don't have that variable
    # quieten logging about empty data frame as it doesn't matter here
    # TODO: make pyam not use the root logger
    logging.getLogger().setLevel(logging.CRITICAL)
    not_to_fill = to_fill_i.filter(variable=req_variable)
    logging.getLogger().setLevel(logging.WARNING)

    to_fill_var = to_fill_i.copy()
    if not not_to_fill.data.empty:
        for (model, scenario), _ in not_to_fill.data.groupby(["model", "scenario"]):
            to_fill_var = to_fill_var.filter(model=model, scenario=scenario, keep=False)
    if not to_fill_var.data.empty:
        interpolated = filler(to_fill_var)
        return interpolated
    return None
