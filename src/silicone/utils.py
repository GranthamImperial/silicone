import datetime as dt
import logging
import os.path

import numpy as np
import pandas as pd
import pyam
import scipy.interpolate
from openscm_units import ScmUnitRegistry
from pint.errors import DimensionalityError

logger = logging.getLogger(__name__)

# initialise our own registry to avoid conflicts
_ur = ScmUnitRegistry()
_ur.add_standards()

"""
Utils contains a number of helpful functions that don't belong elsewhere.
"""


def find_matching_scenarios(
    options_df,
    to_compare_df,
    variable_follower,
    variable_leaders,
    classify_scenarios,
    classify_models=["*"],
    return_all_info=False,
    use_change_not_abs=False,
):
    """
    Groups scenarios and models into different classifications and uses those to
    work out which group contains a trendline most similar to the data. These
    combinations may group several models/scenarios together by means of wild cards.
    Most similar means having the smallest total squared distance between the
    to_compare_df value of variable_follower and the variable_follower values
    interpolated in options_df at the variable_leaders points in to_compare_df, i.e.
    assuming errors only exist in variable_follower.
    In the event of a tie between different scenario/model classifications, it returns the
    scenario/model combination that occurs earlier in the input lists, looping through
    scenarios first.

    Parameters
    ----------
    options_df : :obj:`pyam.IamDataFrame`
        The dataframe containing the data for each scenario and model

    to_compare_df : :obj:`pyam.IamDataFrame`
        The dataframe we wish to find the scenario group closest to. May contain one
        or more scenarios, we minimise the least squared errors for all the data
        colleectively.

    variable_follower : str
        The variable we want to interpolate and compare to the value in to_compare_df

    variable_leaders : list[str]
        The variable(s) we want to use to construct the interpolation
        (e.g. ``["Emissions|CO2"]``). In the event that there are multiple, we
        interpolate with each one separately and minimise the sum of the squared
        errors.

    classify_scenarios : list[str]
        The names of scenarios or groups of scenarios that are possible matches.
        This may have "\\*"s to represent wild cards, hence multiple scenarios will have
        all their data combined to make the interpolator.

    classify_models : list[str]
        The names of models or groups of models that are possible matches.
        This may have "\\*"s to represent wild cards, hence multiple models will have
        all their data combined to make the interpolator.

    return_all_info : bool
        If True, instead of simply returning the strings specifying the closest
        scenario/model match, we return all scenario/model combinations in order of
        preference, along with the rms distance, quantifying the closeness.

    use_change_not_abs : bool
        If True, the code looks for the trend with the closest *derivatives* rather
        than the closest absolute value, i.e. closest trend allowing for an offset.
        This requires data from more than one time.

    Returns
    -------
    if return_all_info == False:

    (string, string)
        Strings specifying the model (first) and scenario (second) classifications
        that best match the data.

    if return_all_info == True:
    dict
        Maps the model and scenario classification strings to the measure of
        closeness.


    Raises
    ------
    ValueError
        Not all required timepoints are present in the database we crunched, we have
         `{dates we have}` but you passed in `{dates we need}`."
    """
    assert all(
        x in options_df.variable for x in [variable_follower] + variable_leaders
    ), "Not all required data is present in compared series"
    assert len(variable_leaders) == 1, "This is only calibrated to work with one leader"
    time_col = options_df.time_col
    assert (
        to_compare_df.time_col == time_col
    ), "The time column in the data to classify does not match the cruncher"
    times_needed = set(to_compare_df.data[time_col])
    if any(x not in options_df.data[time_col].values for x in times_needed):
        raise ValueError(
            "Not all required timepoints are present in the database we "
            "crunched, we have `{}` but you passed in {}.".format(
                list(set(options_df.data[time_col])),
                list(set(to_compare_df.data[time_col])),
            )
        )
    assert (
        len(times_needed) > 1 or use_change_not_abs == False
    ), "We need data from multiple times in order to calculate a difference."
    if to_compare_df.data.empty:
        print("The database being compared is empty")
        return None
    scen_model_rating = {}
    to_compare_db = _make_wide_db(to_compare_df)
    if use_change_not_abs:
        # Set all values to 0 at time 0 to remove any offset
        _remove_t0_from_wide_db(times_needed, to_compare_db)
    to_compare_db = to_compare_db.reset_index()
    for scenario in classify_scenarios:
        for model in classify_models:
            scenario_db = options_df.filter(
                scenario=scenario,
                model=model,
                variable=variable_leaders + [variable_follower],
            )
            if scenario_db.data.empty:
                scen_model_rating[model, scenario] = np.inf
                logger.warning(
                    "Data with scenario {} and model {} not found in data".format(
                        scenario, model
                    )
                )
                continue

            wide_db = _make_wide_db(scenario_db)
            if use_change_not_abs:
                # Set all values to 0 at time 0
                _remove_t0_from_wide_db(times_needed, wide_db)
            squared_dif = 0
            for leader in variable_leaders:
                all_interps = _make_interpolator(
                    variable_follower, leader, wide_db, time_col
                )
                for row in to_compare_db.iterrows():
                    squared_dif += (
                        row[1][variable_follower]
                        - all_interps[row[1][time_col]](row[1][leader])
                    ) ** 2
            scen_model_rating[model, scenario] = squared_dif
    ordered_scen = sorted(scen_model_rating.items(), key=lambda item: item[1])
    if return_all_info:
        return ordered_scen
    return ordered_scen[0][0]


def _remove_t0_from_wide_db(times_needed, _db):
    """
    This function finds the first set of values for each model and scenario and
    subtracts them from all values to remove the offset.
    """

    for model, scenario in set(
        zip(_db.index.get_level_values("model"), _db.index.get_level_values("scenario"))
    ):
        offset = _db.loc[model, scenario, min(times_needed)].copy().values.squeeze()
        for time in times_needed:
            _db.loc[model, scenario, time] = _db.loc[model, scenario, time] - offset


def _make_interpolator(variable_follower, variable_leader, wide_db, time_col):
    """
    Constructs a linear interpolator for variable_follower as a function of
    (one) variable_leader for each timestep in the data.
    """
    derived_relationships = {}
    for db_time, dbtdf in wide_db.groupby(time_col):
        xs = dbtdf[variable_leader].values.squeeze()
        ys = dbtdf[variable_follower].values.squeeze()
        if xs.shape != ys.shape:
            raise NotImplementedError(
                "Having more than one `variable_leaders` is not yet implemented"
            )
        if not xs.shape:
            # 0D-array, so we can return a single value
            xs = np.array([xs])
            ys = np.array([ys])
        # Ensure that any duplicates are replaced by their average value
        xs_pandas = pd.Series(xs)
        for x_dup in xs_pandas[xs_pandas.duplicated()]:
            inds = np.asarray(xs == x_dup).nonzero()[0]
            ys[inds[0]] = ys[inds].mean()
            xs = np.delete(xs, inds[1:])
            ys = np.delete(ys, inds[1:])
        xs, ys = map(np.array, zip(*sorted(zip(xs, ys))))
        if xs.shape == (1,):
            # If there is only one point, we must duplicate the data for interpolate
            xs = np.append(xs, xs)
            ys = np.append(ys, ys)
        derived_relationships[db_time] = scipy.interpolate.interp1d(
            xs, ys, bounds_error=False, fill_value=(ys[0], ys[-1]), assume_sorted=True
        )
    return derived_relationships


def _make_wide_db(use_db):
    """
    Converts an IamDataFrame into a pandas DataFrame that describes the timeseries
    of variables in index-labelled values.
    """
    idx = ["model", "scenario", use_db.time_col]
    assert (
        use_db.data.groupby(idx + ["variable"]).count().max().max() <= 1
    ), "The table contains multiple entries with the same model and scenario"
    use_db = use_db.pivot_table(index=idx, columns="variable", aggfunc="sum")
    # make sure we don't have empty strings floating around (pyam bug?)
    use_db = use_db.applymap(lambda x: np.nan if isinstance(x, str) else x)
    use_db = use_db.dropna(axis=0)
    return use_db


def _get_unit_of_variable(df, variable, multiple_units="raise"):
    """
    Get the unit of a variable in ``df``

    Parameters
    ----------
    variable : str
        String to use to filter variables

    multiple_units : str
        If ``"raise"``, check that the variable only has one unit and raise an
        ``AssertionError`` if it has more than one unit.

    Returns
    -------
    list
        List of units for the variable

    Raises
    ------
    AssertionError
        ``multiple_units=="raise"`` and the filter results in more than one unit
    """
    units = df.filter(variable=variable).data["unit"].unique()
    if multiple_units == "raise":
        if len(units) > 1:
            raise AssertionError("`{}` has multiple units".format(variable))
        return units

    return units


def return_cases_which_consistently_split(
    df, aggregate, components, how_close=None, metric_name="AR5GWP100"
):
    """
    Returns model-scenario tuples which correctly split up the to_split into the various
    components. Components may contain wildcard "\\*"s to match several variables.

    Parameters
    ----------
    df: :obj:`pyam.IamDataFrame`
        The input dataframe.

    aggregate : str
        Name of the variable that should split into the others

    components : list[str]
        List of the variable names whose sum should equal the to_split value (if
        expressed in common units).

    how_close : dict
        This is a dictionary of numpy.isclose options specifying how exact the match
        must be for the case to be included as passing. By default we specify a relative
        tolerance of 1% ('rtol': 1e-2). The syntax for this can be found in the numpy
        documentation.

    metric_name : str
        The name of the conversion metric to use. This will usually be AR<4/5/6>GWP100.

    Returns
    -------
    list[(str, str, str)]
        List of consistent (Model name, scenario name, region name) tuples.
    """
    if not how_close:
        how_close = {"equal_nan": True, "rtol": 1e-02}
    valid_model_scenario = []
    df = convert_units_to_MtCO2_equiv(
        df.filter(variable=[aggregate] + components), metric_name
    )
    combinations = df.data[["model", "scenario", "region"]].drop_duplicates()
    for ind in range(len(combinations)):
        model, scenario, region = combinations.iloc[ind]
        model_df = df.filter(model=model, scenario=scenario, region=region)
        # The following will often release a warning for empty data
        logging.getLogger("pyam.core").setLevel(logging.CRITICAL)
        to_split_df = model_df.filter(variable=aggregate)
        logging.getLogger("pyam.core").setLevel(logging.WARNING)
        if to_split_df.data.empty:
            continue
        sum_all = model_df.data.groupby(model_df.time_col).agg("sum")
        sum_to_split = to_split_df.data.groupby(model_df.time_col).agg("sum")
        if all(
            [
                np.isclose(
                    sum_all.loc[time, "value"],
                    sum_to_split.loc[time, "value"] * 2,
                    **how_close,
                )
                for time in sum_to_split.index
            ]
        ):
            valid_model_scenario.append((model, scenario, region))
    return valid_model_scenario


def convert_units_to_MtCO2_equiv(df, metric_name="AR5GWP100"):
    """
    Converts the units of gases reported in kt into Mt CO2 equivalent per year

    Uses GWP100 values from either (by default) AR5 or AR4 IPCC reports.

    Parameters
    ----------
    df : :obj:`pyam.IamDataFrame`
        The input dataframe whose units need to be converted.

    metric_name : str
        The name of the conversion metric to use. This will usually be AR<4/5/6>GWP100.

    Return
    ------
    :obj:`pyam.IamDataFrame`
        The input data with units converted.
    """
    # Check things need converting
    convert_to_str = "Mt CO2-equiv/yr"
    convert_to_str_clean = "Mt CO2/yr"
    if df["unit"].isin([convert_to_str, convert_to_str_clean]).all():
        return df

    to_convert_df = df.copy()
    to_convert_var = (
        to_convert_df.filter(unit=[convert_to_str, convert_to_str_clean], keep=False)
        .data[["variable", "unit"]]
        .drop_duplicates()
    )
    to_convert_units = to_convert_var["unit"]
    to_convert_units_clean = {
        unit: unit.replace("-equiv", "").replace("equiv", "")
        for unit in to_convert_units
    }

    conversion_factors = {}
    not_found = []
    with _ur.context(metric_name):
        for unit in to_convert_units:
            if unit in conversion_factors:
                continue

            clean_unit = to_convert_units_clean[unit]
            try:
                conversion_factors[unit] = (
                    _ur(clean_unit).to(convert_to_str_clean).magnitude
                )
            except DimensionalityError:
                raise ValueError(
                    "Cannot convert from {} (cleaned is: {}) "
                    "to {} (cleaned is: {})".format(
                        unit, clean_unit, convert_to_str, convert_to_str_clean
                    )
                )

    assert not not_found, "Not all units can be converted. We lack {}".format(not_found)

    for unit in to_convert_units:
        to_convert_df.convert_unit(
            current=unit,
            to=convert_to_str,
            factor=conversion_factors[unit],
            inplace=True,
        )

    return to_convert_df


def download_or_load_sr15(filename, valid_model_ids="*"):
    """
    Load SR1.5 data, if it isn't there, download it

    Parameters
    ----------
    filename : str
        Filename in which to look for/save the data
    valid_model_ids : str
        Models to return from date

    Returns
    -------
    :obj: `pyam.IamDataFrame`
        The loaded data
    """
    if not os.path.isfile(filename):
        get_sr15_scenarios(filename, valid_model_ids)
    return pyam.IamDataFrame(filename).filter(model=valid_model_ids)


def get_sr15_scenarios(output_file, valid_model_ids):
    """
    Collects world-level data from the IIASA database for the named models and saves
    them to a given location.

    Parameters
    ----------
    output_file : str
        File name and location for data to be saved

    valid_model_ids : list[str]
        Names of models that are to be fetched.
    """
    conn = pyam.iiasa.Connection("IXSE_SR15")
    variables_to_fetch = ["Emissions*"]
    for model in valid_model_ids:
        print("Fetching data for {}".format(model))
        for variable in variables_to_fetch:
            print("Fetching {}".format(variable))
            var_df = conn.query(
                model=model, variable=variable, region="World", timeslice=None
            )
            try:
                df.append(var_df, inplace=True)
            except NameError:
                df = pyam.IamDataFrame(var_df)

    print("Writing to {}".format(output_file))
    df.to_csv(output_file)


def _adjust_time_style_to_match(in_df, target_df):
    if in_df.time_col != target_df.time_col:
        in_df = in_df.timeseries()
        if target_df.time_col == "time":
            target_df_year_map = {v.year: v for v in target_df.timeseries().columns}
            in_df.columns = in_df.columns.map(
                lambda x: target_df_year_map[x]
                if x in target_df_year_map
                else dt.datetime(x, 1, 1)
            )
        else:
            in_df.columns = in_df.columns.map(lambda x: x.year)
        return pyam.IamDataFrame(in_df)

    return in_df


def _remove_equivs(string_to_fix):
    """
    Removes the substring "-equiv" from strings. For use in unit conversion
    Parameter
    ---------
    string_to_fix: str
        The string to strip of "-equiv".
    Returns
    -------
    str
    """
    return string_to_fix.replace("-equiv", "")


def _construct_consistent_values(aggregate_name, components, db_to_generate):
    """
    Calculates the sum of the components and creates an IamDataFrame with this
    value under variable type `aggregate_name`.

    Parameters
    ----------
    aggregate_name : str
        The name of the aggregate variable.

    components : [str]
        List of the names of the variables to be summed.

    db_to_generate : :obj:`pyam.IamDataFrame`
        Input data from which to construct consistent values.

    Returns
    -------
    :obj:`pyam.IamDataFrame`
        Consistently calculated aggregate data.
    """
    assert (
        aggregate_name not in db_to_generate.variable
    ), "We already have a variable of this name"
    relevant_db = db_to_generate.filter(variable=components)
    units = relevant_db.data["unit"].drop_duplicates().sort_values()
    unit_equivs = units.map(lambda x: x.replace("-equiv", "")).drop_duplicates()
    if len(unit_equivs) == 0:
        raise ValueError(
            "Attempting to construct a consistent {} but none of the components "
            "present".format(aggregate_name)
        )
    elif len(unit_equivs) > 1:
        raise ValueError(
            "Too many units found to make a consistent {}".format(aggregate_name)
        )
    use = (
        relevant_db.data.groupby(["model", "scenario", "region", relevant_db.time_col])
        .agg("sum")
        .reset_index()
    )
    # Units are sorted in alphabetical order so we choose the first to get -equiv
    use["unit"] = units.iloc[0]
    use["variable"] = aggregate_name
    return pyam.IamDataFrame(use)


def _make_weighting_series(df, weights):
    """
    Makes a complete list of weights for all scenarios from a dictionary of only
    specific weights.

    Parameters
    ----------
    df: :obj:`pd.DataFrame`
        The timseries with the full set of models and scenarios whose weights
        should be returned
    weights: Dict{(str, str) : float}
        The dictionary, mapping the (model and scenario) tuple onto the weight (relative
        to a weight of 1 for the default). This does not have to include all scenarios
        in df, but cannot include scenarios not in df.

    Returns
    -------
    :obj:`pd.Series`
        A series with index corresponding to the timeseries of the dataframe and values
        corresponding to the weights.

    """
    all_mod_scen = [(i[0], i[1]) for i in df.index.drop_duplicates()]

    if any([key not in all_mod_scen for key in weights.keys()]):
        raise ValueError(
            "Not all the weighting values are found in the database. We "
            "lack {}".format([key for key in weights.keys() if key not in all_mod_scen])
        )
    result = pd.Series(np.ones_like(len(df)), index=df.index)
    for (key, val) in weights.items():
        result[key[0], key[1]] = val
    return result
