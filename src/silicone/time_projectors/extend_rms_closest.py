"""
Module for the database cruncher that uses the rms closest extension method
"""
import logging

import numpy as np
import pandas as pd
from pyam import IamDataFrame

logger = logging.getLogger(__name__)


class ExtendRMSClosest:
    """
    Time projector which extends the timeseries of a variable with
    future timesteps infilled using the values from the 'closest'
    pathway in the infilling database.

    We define the closest pathway as the pathway with the smallest
    time-averaged (over the reported time steps) root mean squared
    difference
    """

    def __init__(self, db):
        """
        Initialise the time projector with a database that contains
        the range of times you wish to see in the output.

        Parameters
        ----------
        db : IamDataFrame
            The database to use
        """
        self._db = db.copy()

    def derive_relationship(self, variable):
        """
        Derives the values for the model/scenario combination in the database
        with the least RMS error.

        Parameters
        ----------
        variable : str
            The variable for which we want to calculate the timeseries (e.g.
            `Emissions|CO2`).

        Returns
        -------
        :obj: `pyam.IamDataFrame`
            Filled in data (without original source data)
        """
        iamdf = self._get_iamdf_variable(variable)

        infiller_time_col = iamdf.time_col
        data_follower_unit = iamdf.data["unit"].unique()

        assert (
            len(data_follower_unit) == 1
        ), "The infiller database has {} units in it. It should have one".format(
            len(data_follower_unit)
        )

        def filler(in_iamdf):
            """
            Filler function

            Parameters
            ----------
            in_iamdf : pyam.IamDataFrame
                Input data to be infilled

            Returns
            -------
            :obj:pyam.IamDataFrame
                Filled in data (without original source data)

            Raises
            ------
            ValueError
                "The infiller database does not extend in time past the target "
                "database, so no infilling can occur."
            """
            target_df = in_iamdf.filter(variable=variable)
            if target_df.empty:
                error_msg = "No data for `variable`({}) in target dataframe".format(
                    variable
                )
                raise ValueError(error_msg)

            key_timepoints = target_df.data[infiller_time_col]
            later_times = [
                t
                for t in iamdf.data[infiller_time_col].unique()
                if t > max(key_timepoints)
            ]

            if not later_times:
                raise ValueError(
                    "The infiller database does not extend in time past the target"
                    "database, so no infilling can occur"
                )
            key_timepoint_filter = {infiller_time_col: key_timepoints}

            def get_values_at_key_timepoints(idf, time_filter):
                to_return = idf.filter(**time_filter)
                if to_return.data.empty:
                    raise ValueError(
                        "Not timeseries overlap between original and unfilled data"
                    )

                return to_return.timeseries()

            infiller_at_key_times = get_values_at_key_timepoints(
                iamdf, key_timepoint_filter
            )
            target_at_key_times = get_values_at_key_timepoints(
                target_df, key_timepoint_filter
            )

            closest_model, closest_scenario = _select_closest(
                infiller_at_key_times, target_at_key_times
            )
            tmp = iamdf.filter(
                model=closest_model, scenario=closest_scenario
            ).timeseries()
            output_ts = target_df.timeseries()
            for time in later_times:
                output_ts[time] = tmp[time].values[0]
            for col in output_ts.columns:
                if col not in later_times:
                    del output_ts[col]
            return IamDataFrame(output_ts)

        return filler

    def _get_iamdf_variable(self, variable):
        if variable not in self._db.variable:
            error_msg = "No data for `variable`({}) in database".format(variable)
            raise ValueError(error_msg)

        return self._db.filter(variable=variable)


def _select_closest(to_search_df, target_df):
    if target_df.shape[1] != to_search_df.shape[1]:
        raise ValueError(
            "Target array does not match the size of the searchable arrays"
        )

    rms = pd.Series(index=to_search_df.index, dtype=np.float64)
    target_for_var = {}
    for var in to_search_df.index.get_level_values("variable").unique():
        target_for_var[var] = target_df[
            target_df.index.get_level_values("variable") == var
        ].squeeze()
    var_index = to_search_df.index.names.index("variable")
    for label, row in to_search_df.iterrows():
        varname = label[var_index]
        rms.loc[label] = (((target_for_var[varname] - row) ** 2).mean()) ** 0.5
    rmssums = rms.groupby(level=["model", "scenario"], sort=False).sum()
    return rmssums.idxmin()
