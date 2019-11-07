"""
Module for the database cruncher which uses the 'time-dependent ratio' technique.
"""
import warnings

import numpy as np
from pyam import IamDataFrame

from .base import _DatabaseCruncher


class DatabaseCruncherTimeDepRatio(_DatabaseCruncher):
    """
    Database cruncher which uses the 'time-dependent ratio' technique.

    This cruncher derives the relationship between two variables by simply assuming
    that the follower timeseries is equal to the lead timeseries multiplied by a
    time-dependent scaling factor. The scaling factor is the ratio of the
    follower variable to the lead variable.

    Once the relationship is derived, the 'filler' function will infill following:

    .. math::
        E_f(t) = s(t) * E_l(t)

    where :math:`E_f(t)` is emissions of the follower variable and :math:`E_l(t)` is
    emissions of the lead variable.

    :math:`s` is the scaling factor, calculated as

    .. math::
        s(t) = \\frac{ E_f(t) }{ E_l(t) }

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
        data_follower = iamdf_follower.data

        data_follower_unit = data_follower["unit"].values.unique
        if data_follower_unit.size == 1:
            data_follower_unit = data_follower_unit[0]
        else:
            raise ValueError("Multiple units in follower data")
        data_follower_time_col = iamdf_follower.time_col


        def filler(in_iamdf, interpolate=False):
            """
            Filler function derived from :obj:`DatabaseCruncherTimeDepRatio`.

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

            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )
            times_needed = set(in_iamdf.data[in_iamdf.time_col])
            if any(times_needed not in iamdf_follower[
                data_follower_time_col
            ]):
                error_msg = (
                    "Not all required timepoints are in the data for the lead gas ({})"
                    .format(
                        variable_leaders[0]
                    )
                )
                raise ValueError(error_msg)

            scaling = in_iamdf.filter(variable=variable_leaders)["value"] / \
                data_follower.filter(variable=variable_leaders, year=times_needed)["value"]
            output_ts = (lead_var.timeseries().T * scaling).T.reset_index()

            output_ts["variable"] = variable_follower
            output_ts["unit"] = data_follower_unit

            return IamDataFrame(output_ts)

        return filler

    def _get_iamdf_follower(self, variable_follower, variable_leaders):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherLeadGas`, ``variable_leaders`` should only "
                "contain one variable"
            )

        self._check_follower_and_leader_in_db(variable_follower, variable_leaders)

        iamdf_follower = self._db.filter(variable=variable_follower)
        data_follower = iamdf_follower.data
        if data_follower.shape[0] != 1:
            error_msg = "More than one data point for `variable_follower` ({}) in database".format(
                variable_follower
            )
            raise ValueError(error_msg)

        return iamdf_follower
