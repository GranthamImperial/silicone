"""
Module for the database cruncher which uses the 'lead gas' technique.
"""
import warnings

import pyam
from silicone.utils import select_closest

from .base import _DatabaseCruncher


class DatabaseCruncherRMSClosest(_DatabaseCruncher):
    """
    Database cruncher which finds the root mean squared closest path and returns its values.

    This cruncher derives the relationship between two variables by finding the time-averaged root mean squared (L2)
    nearest path in the test database and reporting back the follower data for this trend.
    Paths that do not contain the entirity of the timeseries will not be investigated.
    The analysis requires a precise match of times between the database used to derive the relationship and the one
    used to infill it.

    """

    def derive_relationship(self, variable_follower, variable_leaders, **kwargs):
        """
        Derive the relationship between two variables from the database.

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|C5F12"``).

        variable_leaders : list[str]
            The variable we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ``["CO2"]``).

        **kwargs
            Keyword arguments used by this class to derive the relationship between
            ``variable_follower`` and ``variable_leaders``.

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

        ValueError
            There is more than one value for ``variable_follower`` in the database.
        """
        self._check_iamdf_follower_and_lead(variable_follower, variable_leaders)
        iamdf_follower = self._get_iamdf_section(variable_follower)
        iamdf_lead = self._get_iamdf_section(variable_leaders)
        data_follower = iamdf_follower.data

        data_follower_key_year_val = data_follower["value"].values.squeeze()
        data_follower_unit = data_follower["unit"].values[0]

        data_follower_time_col = iamdf_follower.time_col

        def filler(in_iamdf, interpolate=False):
            """
            Filler function derived from :obj:`DatabaseCruncherRMSClosest`.

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
            lead_var = in_iamdf.filter(variable=variable_leaders)

            # for other crunchers, unit check would look like this (doesn't actually
            # matter for this cruncher)
            # when we do unit conversion we should add OpenSCM as a dependency as it
            # has all the emissions units inbuilt
            """
            var_units = lead_var.variables(True)
            if var_units.shape[0] != 1:
                raise ValueError("More than one unit detected for input timeseries")
            if (
                var_units.set_index("variable").loc[variable_leaders[0]]["unit"]
                != "expected_unit"
            ):
                raise ValueError(
                    "Units of lead variable is meant to be `expected_unit`, found `other_unit`"
                )
            """
            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )

            key_timepoints_filter_iamdf = {
                data_follower_time_col: lead_var[data_follower_time_col]
            }
            key_timepoints_filter_lead = {
                data_follower_time_col: iamdf_lead[data_follower_time_col]
            }

            def get_values_at_key_timepoints(idf, time_filter):
                # filter warning about empty data frame as we handle it ourselves
               try:
                    to_return = idf.filter(**time_filter)
                    if to_return.data.empty:
                        raise ValueError("No time series overlap between the original and unfilled data.")
                    return to_return
               except KeyError as e:
                   raise ValueError("No time series overlap between the original and unfilled data.")

            lead_var_timeseries = get_values_at_key_timepoints(lead_var, key_timepoints_filter_lead)\
                .timeseries().dropna()
            iamf_lead_timeseries = get_values_at_key_timepoints(iamdf_lead, key_timepoints_filter_iamdf)\
                .timeseries().dropna()
            if lead_var_timeseries.empty or iamf_lead_timeseries.empty:
                raise ValueError("No time series overlap between the original and unfilled data.")
            closest_index = []
            output_ts_list = []
            for row in range(lead_var_timeseries.shape[1]):
                closest_index.append(select_closest(iamf_lead_timeseries, lead_var_timeseries.iloc[row]))
                # Filter to find the matching follow data for the same model, scenario and region
                tmp = iamdf_follower.filter(closest_index[row][0:3])
                # Update the model and scenario to match the elements of the input.
                tmp['model'] = lead_var_timeseries.index[row][0]
                tmp['scenario'] = lead_var_timeseries.index[row][1]
                output_ts_list.append(tmp)

            return pyam.concat(output_ts_list)

        return filler

    def _check_iamdf_follower_and_lead(self, variable_follower, variable_leaders):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherRMSClosest`, ``variable_leaders`` should only "
                "contain one variable"
            )

        if not all([v in self._db.variables().tolist() for v in variable_leaders]):
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

