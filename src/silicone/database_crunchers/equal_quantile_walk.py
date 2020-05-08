"""
Module for the database cruncher which uses the 'equal quantile walk' technique.
"""

import numpy as np
import scipy.interpolate
from pyam import IamDataFrame

from .base import _DatabaseCruncher


class EqualQuantileWalk(_DatabaseCruncher):
    """
    Database cruncher which uses the 'equal quantile walk' technique.

    This cruncher assumes that the amount of effort going into reducing one emission set
    is equal to that for another emission, therefore the lead and follow data should be
    at the same quantile of all pathways in the infiller database.
    It calculates the quantile of the lead infillee data in the lead infiller database,
    then outputs that quantile of the follow data in the infiller database.
    """

    def derive_relationship(self, variable_follower, variable_leaders):
        """
        Derive the relationship between two variables from the database.

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|C5F12"``).

        variable_leaders : list[str]
            The variable we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ``["Emissions|CO2"]``).

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
        iamdf_follower = self._get_iamdf_follower(variable_follower, variable_leaders)
        follower_ts = iamdf_follower.timeseries()

        data_follower_time_col = iamdf_follower.time_col
        data_follower_unit = iamdf_follower["unit"].values[0]
        lead_ts = self._db.filter(variable=variable_leaders).timeseries()
        lead_unit = lead_ts.index.get_level_values("unit")[0]

        def filler(in_iamdf, interpolate=False):
            """
            Filler function derived from :obj:`LatestTimeRatio`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            interpolate : bool
                If the key year for filling is not in ``in_iamdf``, should a value be
                interpolated?

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled in data (without original source data)

            Raises
            ------
            ValueError
                The key year for filling is not in ``in_iamdf`` and ``interpolate is
                False``.
            """
            lead_in = in_iamdf.filter(variable=variable_leaders)
            if not all(lead_in.variables(True)["unit"] == lead_unit):
                raise ValueError(
                    "Units of lead variable is meant to be `{}`, found `{}`".format(
                        lead_unit, lead_in.variables(True)["unit"].tolist()
                    )
                )

            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )
            if lead_in.data.empty:
                raise ValueError(
                    "There is no data for {} so it cannot be infilled".format(
                        variable_leaders
                    )
                )
            output_ts = lead_in.timeseries()
            if any(
                [
                    (time not in lead_ts.columns) or (time not in follower_ts.columns)
                    for time in output_ts.columns
                ]
            ):
                # We allow for cases where either lead or follow have gaps
                raise ValueError(
                    "Not all required timepoints are present in the database we "
                    "crunched, we crunched \n\t{} for the lead and \n\t{} for the "
                    "follow \nbut you passed in \n\t{}".format(
                        lead_ts.columns, follower_ts.columns, output_ts.columns
                    )
                )
            for col in output_ts.columns:
                output_ts[col] = self._find_same_quantile(
                    follower_ts[col], lead_ts[col], output_ts[col]
                )
            output_ts = output_ts.reset_index()
            output_ts["variable"] = variable_follower
            output_ts["unit"] = data_follower_unit
            return IamDataFrame(output_ts)

        return filler

    def _get_iamdf_follower(self, variable_follower, variable_leaders):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `EqualQuantileWalk`, ``variable_leaders`` should only "
                "contain one variable"
            )

        self._check_follower_and_leader_in_db(variable_follower, variable_leaders)

        return self._db.filter(variable=variable_follower)

    def _find_same_quantile(self, follow_vals, lead_vals, lead_input):
        # Dispose of nans that can cloud the calculation
        follow_vals = follow_vals[~np.isnan(follow_vals)]
        lead_vals = lead_vals[~np.isnan(lead_vals)]
        len_lead_not_nan = len(lead_vals)
        if len_lead_not_nan <= 1:
            # If there is only a single value we have to return the best guess.
            return np.nanmean(follow_vals)
        lead_vals = lead_vals.sort_values()
        quant_of_lead_vals = np.arange(len_lead_not_nan) / (len_lead_not_nan - 1)
        if any(quant_of_lead_vals > 1) or any(quant_of_lead_vals < 0):
            raise ValueError("Impossible quantiles!")
        input_quantiles = scipy.interpolate.interp1d(
            lead_vals, quant_of_lead_vals, bounds_error=False, fill_value=(0, 1)
        )(lead_input)
        return np.nanquantile(follow_vals, input_quantiles)
