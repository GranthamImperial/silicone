import logging
import warnings

import pandas as pd
import pyam
import tqdm

from silicone.database_crunchers import ConstantRatio, QuantileRollingWindows

"""
Infills all required data for MAGICC and FAIR emulators with minimal configuration 
"""


def infill_all_required_variables(
    to_fill,
    database,
    variable_leaders,
    required_variables_list=None,
    cruncher=QuantileRollingWindows,
    output_timesteps=None,
    infilled_data_prefix=None,
    to_fill_old_prefix=None,
    check_data_returned=False,
    **kwargs,
):
    """
    This is function designed to infill all required data given a minimal amount of
    input.

    Parameters
    ----------
    to_fill : :obj:`pyam.IamDataFrame`
        The dataframe which is to be infilled

    database: :obj:`pyam.IamDataFrame`
        The dataframe containing all information to be used in the infilling process.

    variable_leaders: list[str]
        The name of the variable(s) found in to_fill which should be used to determine
        the values of the other variables. For most infillers (including the default)
        this list must contain only one entry. E.g. ["Emissions|CO2"]

    required_variables_list: list[str]
        The list of variables to infill. Each will be done separately. The default
        behaviour (None option) will result in this being filled with the complete list
        of required emissions.

    cruncher : :class:
        The class of cruncher to use to compute the infilled values. Defaults to
        QuantileRollingWindows, which uses the median value of a rolling
        window. See the cruncher documentation for more details.

    output_timesteps : list[int or datetime]
        List of times at which to return infilled values. Will interpolate values in
        between known data, but will not extend beyond the range of data provided.

    infilled_data_prefix : str
        A string that should be prefixed on all the variable names of the results
        returned. Used to distinguish returned values from those input.

    to_fill_old_prefix : str
        Any string already found at the beginning of the variables names of the input
        `to_fill` dataframe. This will be removed before comparing the variable names
        with `database`.

    check_data_returned : bool
        If true, we perform checks that all desired data has been returned. Potential
        reasons for failing this include requesting results at times outside our input
        time range, as well as code bugs.

    ** kwargs :
        An optional dictionary of keyword : arguments to be used with the cruncher.

    Returns
    -------
    :obj:`pyam.IamDataFrame`
        The infilled dataframe (including input data) at requested times. All variables
        now begin with infilled_data_prefix instead of to_fill_old_prefix.
    """
    # Use default arguments for unfilled options:
    if output_timesteps is None:
        if to_fill.time_col == "time":
            raise ValueError(
                "No default behaviour for output_timesteps when dataframe has time "
                "column instead of years"
            )
        output_timesteps = [2015] + list(range(2020, 2101, 10))

    if required_variables_list is None:
        required_variables_list = [
            "Emissions|BC",
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
            "Emissions|HFC|HFC245ca",
            "Emissions|HFC|HFC125",
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
        to_fill.rename(
            {
                "variable": {
                    var: var.replace(to_fill_old_prefix + "|", "")
                    for var in to_fill.variable
                }
            },
            inplace=True,
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
    assert len(to_fill.region) == 1, "There are {} regions in the data.".format(
        len(to_fill.region)
    )
    assert len(database.region) == 1
    assert (
        to_fill.data["region"].iloc[0] == database.data["region"].iloc[0]
    ), "The cruncher data and the infilled data have different regions."
    # Perform any interpolations required here
    to_fill_orig = to_fill.copy()
    timecol = database.time_col
    assert timecol == to_fill.time_col
    df_times_missing = database.timeseries().isna().sum() > 0
    to_fill_times_missing = to_fill.timeseries().isna().sum() > 0
    for time in output_timesteps:
        if time not in database[timecol].tolist() or df_times_missing[time]:
            database = database.interpolate(time, inplace=False)
        if time not in to_fill[timecol].tolist() or to_fill_times_missing[time]:
            to_fill = to_fill.interpolate(time, inplace=False)
    # Nans in additional columns break pyam, so we overwrite them
    database.data[database.extra_cols] = database.data[database.extra_cols].fillna(0)
    to_fill.data[to_fill.extra_cols] = to_fill.data[to_fill.extra_cols].fillna(0)
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
    unavailable_variables = [
        variab for variab in required_variables_list if variab not in database.variable
    ]
    if unavailable_variables:
        warnings.warn(
            UserWarning(
                "No data for {}, it will be infilled with 0s".format(
                    unavailable_variables
                )
            )
        )
        # Infill the required variables with 0s.
        kwarg_dict = {"ratio": 0, "units": "Mt CO2-equiv/yr"}
        to_fill = _perform_crunch_and_check(
            unavailable_variables,
            variable_leaders,
            to_fill,
            database,
            ConstantRatio,
            output_timesteps,
            to_fill_orig,
            check_data_returned=False,
            **kwarg_dict,
        )
    available_variables = [
        variab
        for variab in required_variables_list
        if variab not in unavailable_variables
    ]
    if available_variables:
        to_fill = _perform_crunch_and_check(
            available_variables,
            variable_leaders,
            to_fill,
            database,
            cruncher,
            output_timesteps,
            to_fill_orig,
            check_data_returned=check_data_returned,
            **kwargs,
        )
    if infilled_data_prefix:
        to_fill.rename(
            {
                "variable": {
                    var: infilled_data_prefix + "|" + var for var in to_fill.variable
                }
            },
            inplace=True,
        )
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
    **kwargs,
):
    """
    Takes a list of scenarios to infill and infills them according to the options
    presented.

    Parameters
    ----------
    required_variables : list[str]
        The variable names to infill

    leaders : list[str]
        The leaders to guide the infilling

    to_fill : IamDataFrame
        The data frame to infill

    df : IamDataFrame
        The data frame to base the infilling on

    type_of_cruncher : :obj: silicone cruncher
        the silicone package cruncher class to use for the infilling

    output_timesteps : list[int or datetime]
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
    cruncher = type_of_cruncher(df)
    for req_var in tqdm.tqdm(required_variables, desc="Filling required variables"):
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
            if df.time_col == "year":
                assert all(
                    [y in msvdf_data[df.time_col].values for y in output_timesteps]
                ), "We do not have data for all required timesteps"
            else:
                output_timesteps_datetime = pd.to_datetime(output_timesteps)
                assert all(
                    [
                        y in msvdf_data[df.time_col].values
                        for y in output_timesteps_datetime.values
                    ]
                ), "We do not have data for all required timesteps"

    # Check no data was overwritten by accident
    for model in tqdm.tqdm(
        to_fill_orig.model, desc="Consistency with original model data checks"
    ):
        mdf = to_fill_orig.filter(model=model, variable=leaders + required_variables)
        for scenario in mdf.scenario:
            msdf = mdf.filter(scenario=scenario)
            msdf_filled = to_fill.filter(
                model=model, scenario=scenario, variable=msdf["variable"].unique()
            )

            common_times = set(msdf_filled[msdf.time_col]).intersection(
                msdf[msdf.time_col]
            )
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
    """
    A function used to iterate the actual crunching if the data doesn't already
    exist.
    Parameters
    ----------
    cruncher_i : :obj: silicone cruncher
        the initiated silicone cruncher to use for the infilling

    req_variable : str
        The follower variable to infill.

    leader_i : list[str]
        The leader variable to guide the infilling.

    to_fill_i : IamDataFrame
        The dataframe to infill.

    kwargs : Dict
        Any key word arguments to include in the cruncher calculation

    Returns
    -------
    :obj:IamDataFrame
        The infilled component of the dataframe (or None if no infilling done)
    """
    filler = cruncher_i.derive_relationship(req_variable, leader_i, **kwargs)
    # only fill for scenarios who don't have that variable
    # quieten logging about empty data frame as it doesn't matter here
    logging.getLogger("pyam.core").setLevel(logging.CRITICAL)
    not_to_fill = to_fill_i.filter(variable=req_variable)

    to_fill_var = to_fill_i.copy()
    if not not_to_fill.data.empty:
        for (model, scenario), _ in not_to_fill.data.groupby(["model", "scenario"]):
            to_fill_var = to_fill_var.filter(model=model, scenario=scenario, keep=False)
    if not to_fill_var.data.empty:
        interpolated = filler(to_fill_var)
        return interpolated
    logging.getLogger("pyam.core").setLevel(logging.WARNING)
    return None
