"""
Module for the database cruncher which uses the 'closest RMS' technique.
"""
import warnings

import numpy as np
import pandas as pd
import pyam

from .base import _DatabaseCruncher


class RMSClosest(_DatabaseCruncher):
    """
    Database cruncher which uses the 'closest RMS' technkque.

    This cruncher derives the relationship between two or more variables by finding the
    scenario which has the most similar timeseries for the lead gases in the database.
    The follower gas timeseries is then simply copied from the closest scenario.

    Here, 'most similar' is defined as the smallest time-averaged root mean squared (L2)
    difference. If multiple lead values are used, they may be weighted differently to
    account for differences between the reported units. The most similar model/scenario
    combination minimises

    .. math::
        RMS error = \\sum_l w_l \\left ( \\frac{1}{n} \\sum_{t=0}^n (E_l(t) - e_l(t))^2 \\right )^{1/2}

    where :math:`l` is a lead gas, :math:`w_l` is a weighting for that lead gas,
    :math:`n` is the total number of timesteps in all lead gas timeseries,
    :math:`E_l(t)` is the lead gas emissions timeseries and :math:`e_l(t)` is a lead
    gas emissions timeseries in the infiller database.
    """

    def derive_relationship(self, variable_follower, variable_leaders, weighting=None):
        """
        Derive the relationship between the lead and the follow variables from the
        database.

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|C5F12"``).

        variable_leaders : list[str]
            The variable we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ``["Emissions|CO2"]``). This may contain
            multiple elements.

        weighting : dict{str: float}
            When used with multiple lead variables, this weighting factor controls the
            relative importance of different variables for determining closeness. E.g.
            if wanting to compare both CO2 and CH4 emissions reported in mass
            units but weighted by the AR5 GWP100 metric, this would be
            {"Emissions|CO2": 1, "Emissions|CH4": 28}.

        Returns
        -------
        :obj:`func`
            Function which takes a :obj:`pyam.IamDataFrame` containing
            ``variable_leaders`` timeseries and returns timeseries for
            ``variable_follower`` based on the derived relationship between the two.
            Please see the source code for the exact definition (and docstring) of the
            returned function.

        Raises
        ------
        ValueError
            ``variable_leaders`` contains more than one variable.

        ValueError
            There is no data for ``variable_leaders`` or ``variable_follower`` in the
            database.
        """
        self._check_iamdf_lead(variable_leaders)
        iamdf_follower = self._get_iamdf_section(variable_follower)
        data_follower_time_col = iamdf_follower.time_col
        iamdf_lead = self._db.filter(variable=variable_leaders)
        if not weighting:
            weighting = {variab: 1 for variab in variable_leaders}
        if any(var not in weighting.keys() for var in variable_leaders):
            raise ValueError("Weighting does not include all lead variables.")
        iamdf_lead, iamdf_follower = _filter_for_overlap(
            iamdf_lead,
            iamdf_follower,
            ["scenario", "model", data_follower_time_col],
            variable_leaders,
        )

        leader_var_unit = {
            var["variable"]: var["unit"]
            for _, var in iamdf_lead[["variable", "unit"]].drop_duplicates().iterrows()
        }

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`RMSClosest`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled in data (without original source data)

            Raises
            ------
            ValueError
                If there are any inconsistencies between the timeseries, units or
                expectations of the program and ``in_iamdf``, compared to the database
                used to generate this ``filler`` function.
            """
            lead_var = in_iamdf.filter(variable=variable_leaders)

            var_units = lead_var[["variable", "unit"]].drop_duplicates()
            if any([key not in lead_var.variable for key in leader_var_unit.keys()]):
                raise ValueError(
                    "Not all required variables are present in the infillee database"
                )
            if any(
                unit["unit"] != leader_var_unit[unit["variable"]]
                for _, unit in var_units.iterrows()
            ):
                raise ValueError(
                    "Units of lead variable is meant to be {}, found {}".format(
                        leader_var_unit, var_units
                    )
                )

            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )

            key_timepoints_filter_iamdf = {
                data_follower_time_col: list(set(lead_var[data_follower_time_col]))
            }
            key_timepoints_filter_lead = {
                data_follower_time_col: list(set(iamdf_lead[data_follower_time_col]))
            }

            def get_values_at_key_timepoints(idf, time_filter):
                # filter warning about empty data frame as we handle it ourselves
                to_return = idf.filter(**time_filter)
                if to_return.data.empty:
                    raise ValueError(
                        "No time series overlap between the original and unfilled data"
                    )
                return to_return

            lead_var_filt = get_values_at_key_timepoints(
                lead_var, key_timepoints_filter_lead
            )
            lead_var_timeseries = lead_var_filt.timeseries()
            iamdf_lead_timeseries = get_values_at_key_timepoints(
                iamdf_lead, key_timepoints_filter_iamdf
            ).timeseries()

            output_ts_list = []
            for _, (model, scenario) in (
                lead_var_filt.data[["model", "scenario"]].drop_duplicates().iterrows()
            ):
                lead_var_mod_scen = lead_var_timeseries[
                    (lead_var_timeseries.index.get_level_values("model") == model)
                    & (
                        lead_var_timeseries.index.get_level_values("scenario")
                        == scenario
                    )
                ]
                if len(lead_var_mod_scen) != len(variable_leaders):
                    raise ValueError(
                        "Insufficient variables are found to infill model {}, scenario "
                        "{}. Only found {}.".format(model, scenario, lead_var_mod_scen)
                    )
                closest_model, closest_scenario = _select_closest(
                    iamdf_lead_timeseries, lead_var_mod_scen, weighting
                )

                # Filter to find the matching follow data for the same model, scenario
                # and region
                tmp = iamdf_follower.filter(
                    model=closest_model, scenario=closest_scenario
                ).data

                # Update the model and scenario to match the elements of the input.
                tmp["model"] = model
                tmp["scenario"] = scenario
                output_ts_list.append(tmp)
                for col in in_iamdf.extra_cols:
                    tmp[col] = lead_var_mod_scen.index.get_level_values(col).tolist()[0]
            return pyam.concat(output_ts_list)

        return filler

    def _check_iamdf_lead(self, variable_leaders):
        if not all([v in self._db.variable for v in variable_leaders]):
            error_msg = "No data for `variable_leaders` ({}) in database".format(
                variable_leaders
            )
            raise ValueError(error_msg)

    def _get_iamdf_section(self, variables):
        # filter warning about empty data frame as we handle it ourselves
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iamdf_section = self._db.filter(variable=variables)

        data_section = iamdf_section.data
        if data_section.empty:
            error_msg = "No data for `variable_follower` ({}) in database".format(
                variables
            )
            raise ValueError(error_msg)

        return iamdf_section


def _select_closest(to_search_df, target_df, weighting):
    """
    Find model/scenario combo in ``to_search_df`` that is closest to that of the target.

    Here, 'closest' is in the root-mean squared sense. In the event that multiple model/
    scenarios are equally close, returns first row.

    Parameters
    ----------
    to_search_df : :obj:`pd.DataFrame`
        The model/scenario combos to search for the closest case. A timeseries.

    target_df : :obj:`pd.DataFrame`
        The data to which we want to be close. A timeseries.

    weighting : map{str: float}
        Maps the variable name onto the weighting for that variable.

    Returns
    -------
    dict
        Index of the closest timeseries.
    """

    if target_df.shape[1] != to_search_df.shape[1]:
        raise ValueError(
            "Target array does not match the size of the searchable arrays"
        )
    if any(
        x not in target_df.index.get_level_values("variable")
        for x in to_search_df.index.get_level_values("variable")
    ):
        raise ValueError("No variable overlap between target and infiller databases.")
    if any(col not in target_df.columns for col in to_search_df.columns):
        raise ValueError("Time values mismatch between target and infiller databases.")
    rms = pd.Series(index=to_search_df.index, dtype=np.float64)
    target_for_var = {}
    for var in to_search_df.index.get_level_values("variable").unique():
        target_for_var[var] = target_df[
            target_df.index.get_level_values("variable") == var
        ].squeeze()
    var_index = to_search_df.index.names.index("variable")
    for label, row in to_search_df.iterrows():
        varname = label[var_index]
        rms.loc[label] = (
            ((target_for_var[varname] - row) ** 2).mean()
        ) ** 0.5 * weighting[varname]
    rmssums = rms.groupby(level=["model", "scenario"], sort=False).sum()
    return rmssums.idxmin()


def _filter_for_overlap(df1, df2, cols, leaders):
    """
    Returns overlapping model/scenario combinations in the two input dataframes, which
    must have the same columns.
    Parameters
    ----------
    df1 : :obj:`pd.DataFrame`
        The first dataframe (order is irrelevant)
    df2 : :obj:`pd.DataFrame`
        The second dataframe (order is irrelevant)
    cols: list[str]
        List of columns that should be identical between the two dataframes. Typically
        "scenario", "model", and whatever the time column is.
    leaders : list[str]
        List of lead variables that must be found in all acceptable model/scenarios
        combinations.
    Returns
    -------
    (:obj:`pd.DataFrame`, :obj:`pd.DataFrame`)
        The two dataframes in the order they were put in, now filtered for some columns
        being identical.
    """
    lead_data = df1.data.set_index(cols)
    follow_data = df2.data.set_index(cols)
    # We only want to select model/scenario cases where we have data for all leaders
    shared_indices = [
        ind
        for ind in follow_data.index
        if lead_data.index.tolist().count(ind) == len(leaders)
    ]
    # We need to remove duplicates
    shared_indices = list(dict.fromkeys(shared_indices))
    if shared_indices:
        lead_data = lead_data.loc[shared_indices]
        follow_data = follow_data.loc[shared_indices]
        return pyam.IamDataFrame(lead_data), pyam.IamDataFrame(follow_data)
    raise ValueError("No model/scenario overlap between leader and follower data")
