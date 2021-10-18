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

        iamdf_lead_data, iamdf_follower_data = _filter_for_overlap(
            iamdf_lead,
            iamdf_follower,
            ["scenario", "model", data_follower_time_col],
            variable_leaders,
        )

        iamdf_lead_ts = pyam.IamDataFrame(iamdf_lead_data).timeseries()
        iamdf_follower_ts = pyam.IamDataFrame(iamdf_follower_data).timeseries()

        leader_var_unit = {
            var["variable"]: var["unit"]
            for _, var in iamdf_lead_data[["variable", "unit"]]
            .drop_duplicates()
            .iterrows()
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

            var_units = lead_var.data[["variable", "unit"]].drop_duplicates()
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

            (
                iamdf_lead_ts_here,
                lead_var_ts_here,
                common_cols,
            ) = _get_common_timeseries_and_cols(
                iamdf_lead_ts.copy(), lead_var.timeseries()
            )
            iamdf_lead_ts_here = iamdf_lead_ts_here[common_cols]
            lead_var_ts_here = lead_var_ts_here[common_cols]

            rms = _calculate_rms(
                iamdf_lead_ts_here,
                lead_var_ts_here,
                weighting,
            )

            out = _combine_rms_and_database(
                lead_var_ts_here,
                iamdf_follower_ts,
                rms,
                [variable_follower],
            )

            return out

        return filler

    def infill_multiple(
        self, to_infill, variable_followers, variable_leaders, weighting=None
    ):
        """
        Infill multiple variables simultaneously

        This can be much faster as the RMS between the lead and follower
        scenarios only needs to be calculated once.

        Parameters
        ----------
        to_infill : :class:`pyam.IamDataFrame`
            The timeseries to infill

        variable_followers : list[str]
            The variables for which we want to infill timeseries (e.g.
            ``["Emissions|C5F12", "Emissions|C4F10"]``).

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
        :class:`pyam.IamDataFrame`
            Infilled timeseries

        Raises
        ------
        ValueError
            There is no data for ``variable_leaders`` or ``variable_follower`` in the
            database.
        """
        self._check_followers_and_leaders_in_db(variable_followers, variable_leaders)
        db_lead = self._db.filter(variable=variable_leaders)
        to_infill_lead = to_infill.filter(variable=variable_leaders)

        db_lead_ts, to_infill_lead_ts, common_cols = _get_common_timeseries_and_cols(
            db_lead.timeseries(), to_infill_lead.timeseries()
        )

        rms = _calculate_rms(
            db_lead_ts[common_cols], to_infill_lead_ts[common_cols], weighting
        )

        db_timeseries = self._db.filter(variable=variable_followers).timeseries()
        db_timeseries = db_timeseries[common_cols]

        out = _combine_rms_and_database(to_infill_lead_ts, db_timeseries, rms, variable_followers)

        return out

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

    def _check_followers_and_leaders_in_db(self, variable_followers, variable_leaders):
        if not all([v in self._db.variable for v in variable_leaders]):
            error_msg = "No data for `variable_leaders` ({}) in database".format(
                variable_leaders
            )
            raise ValueError(error_msg)

        if not all([v in self._db.variable for v in variable_followers]):
            error_msg = "No data for `variable_followers` ({}) in database".format(
                variable_followers
            )
            raise ValueError(error_msg)


def _get_common_timeseries_and_cols(db_lead, to_infill_lead):
    db_lead.index = db_lead.index.rename(
        ["model_db", "scenario_db"], level=["model", "scenario"]
    )

    to_infill_lead.index = to_infill_lead.index.rename(
        ["model_lead", "scenario_lead"], level=["model", "scenario"]
    )

    common_cols = _get_common_cols(db_lead, to_infill_lead)

    return db_lead, to_infill_lead, common_cols


def _calculate_rms(db_lead_ts, to_infill_lead_ts, weighting):
    # check variables first
    ms_lead = ["model_lead", "scenario_lead"]

    required_vars = set(db_lead_ts.index.get_level_values("variable").unique())
    for (model, scenario), msdf in to_infill_lead_ts.groupby(ms_lead):
        msdf_vars = set(msdf.index.get_level_values("variable").unique())
        if msdf_vars != required_vars:
            raise ValueError(
                f"Insufficient variables are found to infill model {model}, scenario "
                f"{scenario}. Only found {msdf}."
            )

    # remove any database timeseries which have nans
    db_lead_ts = db_lead_ts.dropna()

    # align
    db_lead_ts, to_infill_lead_ts = db_lead_ts.align(to_infill_lead_ts)

    rms = ((db_lead_ts - to_infill_lead_ts) ** 2).mean(axis=1) ** 0.5

    if weighting is not None:
        weighting = pd.Series(weighting)
        weighting.index.names = ["variable"]

        rms = rms.multiply(weighting)

    rms = rms.groupby(["model_lead", "scenario_lead", "model_db", "scenario_db"]).sum()

    return rms


def _combine_rms_and_database(to_infill_lead_ts, db_timeseries, rms, variable_followers):
    out = []
    for (model, scenario), to_infill_lead_ts_mod_scen in to_infill_lead_ts.groupby(
        ["model_lead", "scenario_lead"]
    ):
        variable_followers_h = set(variable_followers)

        rms_mod_scen = rms.loc[
            (rms.index.get_level_values("model_lead") == model)
            & (rms.index.get_level_values("scenario_lead") == scenario)
            ,
            :
        ]
        for (model_db, scenario_db), _ in (
            rms_mod_scen[(model, scenario)].sort_values().iteritems()
        ):
            infill_timeseries = db_timeseries.loc[
                (db_timeseries.index.get_level_values("model") == model_db)
                & (db_timeseries.index.get_level_values("scenario") == scenario_db)
                & (
                    db_timeseries.index.get_level_values("variable").isin(
                        variable_followers_h
                    )
                ),
                :,
            ].copy()

            variable_followers_h = variable_followers_h - set(
                infill_timeseries.index.get_level_values("variable")
            )

            infill_timeseries = infill_timeseries.reset_index()
            infill_timeseries.loc[:, "model"] = model
            infill_timeseries.loc[:, "scenario"] = scenario

            for idx_level in to_infill_lead_ts.index.names:
                if idx_level in infill_timeseries or idx_level in ["model_lead", "scenario_lead"]:
                    continue

                val_to_use = to_infill_lead_ts_mod_scen.index.get_level_values(idx_level).unique()
                if len(val_to_use) != 1:
                    raise AssertionError(
                        f"Ambiguous value to use for column {idx_level}, found {val_to_use}"
                    )

                infill_timeseries[idx_level] = val_to_use[0]

            out.append(infill_timeseries)

            if not variable_followers_h:
                break

    out = pyam.IamDataFrame(pd.concat(out))

    return out


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
    shared_indices = lead_data.index[
        lead_data.index.isin(follow_data.index)
    ].value_counts()
    shared_indices = shared_indices[shared_indices == len(leaders)].index.tolist()

    if not shared_indices:
        raise ValueError("No model/scenario overlap between leader and follower data")

    lead_data = lead_data.loc[shared_indices]
    follow_data = follow_data.loc[shared_indices]

    return lead_data.reset_index(), follow_data.reset_index()


def _get_common_cols(lead, base):
    common_cols = lead.columns.intersection(base.columns)
    if common_cols.empty:
        raise ValueError(
            "No time series overlap between the original and unfilled data"
        )

    return common_cols
