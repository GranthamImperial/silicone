"""
Module for the database cruncher which uses the 'closest RMS' technique.
"""
import warnings

import pyam

from .base import _DatabaseCruncher

# TODO: add `interpolate` to filler

class DatabaseCruncherRMSClosest(_DatabaseCruncher):
    """
    Database cruncher which uses the 'closest RMS' technkque.

    This cruncher derives the relationship between two variables by finding the
    scenario which has the closest lead gas timeseries in the database.  The follower
    gas timeseries is then simply copied from the closest scenario.

    Here, 'closest' is defined as the smallest time-averaged root mean squared (L2)
    difference.

    .. math::
        RMS = \\left ( \\frac{1}{n} \\sum_{t=0}^n (E_l(t) - E_l^{d}(t))^2 \\right )^{1/2}

    where :math:`n` is the total number of timesteps in the lead gas' timeseries,
    :math:`E_l(t)` is the lead gas emissions timeseries and :math:`E_l^d(t)` is a lead
    gas emissions timeseries in the database.
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
        self._check_iamdf_lead(variable_leaders)
        iamdf_follower = self._get_iamdf_section(variable_follower)
        iamdf_lead = self._db.filter(variable=variable_leaders)

        leader_unit = iamdf_lead["unit"].unique()
        assert len(leader_unit) == 1  # TODO: use functionality in #13
        leader_unit = leader_unit[0]

        data_follower_time_col = iamdf_follower.time_col

        def filler(in_iamdf, interpolate=False):
            """
            Filler function derived from :obj:`DatabaseCruncherRMSClosest`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            interpolate : bool
                If the lead timeseries years don't match those in the database, should
                we interpolate the database timeseries or simply discard them? TODO:
                test this.

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled in data (without original source data)

            Raises
            ------
            ValueError
                If there are any inconsistencies between the timeseries, units or expectations of the program and the
                in_amdf, compared to the database passed to the cruncher, one of many errors are thrown.
            """
            lead_var = in_iamdf.filter(variable=variable_leaders)

            # when we do unit conversion we should add OpenSCM as a dependency as it
            # has all the emissions units inbuilt

            var_units = lead_var["unit"].unique()
            if len(var_units) != 1:
                raise ValueError("More than one unit detected for input timeseries")

            var_units = var_units[0]
            if var_units != leader_unit:
                raise ValueError(
                    "Units of lead variable is meant to be {}, found {}".format(leader_unit, var_units)
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
                    raise ValueError("No time series overlap between the original and unfilled data")
                return to_return

            lead_var_timeseries = get_values_at_key_timepoints(
                lead_var, key_timepoints_filter_lead
            ).timeseries()
            iamdf_lead_timeseries = get_values_at_key_timepoints(
                iamdf_lead, key_timepoints_filter_iamdf
            ).timeseries()

            output_ts_list = []
            for label, row in lead_var_timeseries.iterrows():
                closest_ts = _select_closest(
                    iamdf_lead_timeseries,
                    row
                )

                # Filter to find the matching follow data for the same model, scenario
                # and region
                tmp = iamdf_follower.filter(
                    model=closest_ts["model"], scenario=closest_ts["scenario"]
                ).data

                # Update the model and scenario to match the elements of the input.
                tmp['model'] = label[lead_var_timeseries.index.names.index("model")]
                tmp['scenario'] = label[lead_var_timeseries.index.names.index("scenario")]
                output_ts_list.append(tmp)

            return pyam.concat(output_ts_list)

        return filler

    def _check_iamdf_lead(self, variable_leaders):
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

def _select_closest(to_search_df, target_series):
    """
    Find row in ``to_search_df`` that is closest to the target array.

    Here, 'closest' is in the root-mean squared sense. In the event that multiple rows
    are equally close, returns first row.

    Parameters
    ----------
    to_search_df : :obj:`pd.DataFrame`
        The rows of this dataframe are the candidate closest vectors

    target_series : :obj:`pd.Series`
        The vector to which we want to be close

    Returns
    -------
    dict
        Metadata of the closest row.
    """
    if len(target_series.shape) != 1:
        raise ValueError("Target array is multidimensional")

    if target_series.shape[0] != to_search_df.shape[1]:
        raise ValueError(
            "Target array does not match the size of the searchable arrays"
        )

    closeness = []
    for label, row in to_search_df.iterrows():
        rms = (((target_series - row) ** 2).mean()) ** 0.5
        closeness.append((label, rms))

    # Find the minimum closeness and return the index of it
    labels, rmss = list(zip(*closeness))
    to_return = rmss.index(min(rmss))
    return dict(zip(to_search_df.index.names, labels[to_return]))
