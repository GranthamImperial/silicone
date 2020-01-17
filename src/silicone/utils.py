import numpy as np
import pandas as pd
import scipy.interpolate

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
        This may have *s to represent wild cards, hence multiple scenarios will have
        all their data combined to make the interpolator.

    classify_models : list[str]
        The names of models or groups of models that are possible matches.
        This may have *s to represent wild cards, hence multiple models will have
        all their data combined to make the interpolator.

    return_all_info : bool
        If True, instead of simply returning the strings specifying the closest
        scenario/model match, we return all scenario/model combinations in order of
        preference, along with

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
        x in options_df.variables().values
        for x in [variable_follower] + variable_leaders
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
                print(
                    "Warning: data with scenario {} and model {} not found in data".format(
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
        use_db.data.groupby(idx + ["variable"]).count()._get_values.max() <= 1
    ), "The table contains multiple entries with the same model and scenario"
    use_db = use_db.pivot_table(index=idx, columns="variable", aggfunc="sum")
    # make sure we don't have empty strings floating around (pyam bug?)
    use_db = use_db.applymap(lambda x: np.nan if isinstance(x, str) else x)
    use_db = use_db.dropna(axis=0)
    return use_db


# TODO: put this in pyam
def _get_unit_of_variable(df, variable, multiple_units="raise"):
    """
    Get the unit of a variable in ``df``

    Parameters
    ----------
    variable : str
        String to use to filter variables

    multiple_units : str
        If ``"raise"``, check that the variable only has one unit and raise an ``AssertionError`` if it has more than one unit.

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
