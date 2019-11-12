"""
Module for the database cruncher which uses the 'time-dependent ratio' technique.
"""
import warnings

import numpy as np
import silicone.stats
from pyam import IamDataFrame

from .base import _DatabaseCruncher


class DatabaseCruncherTimeDepRatio(_DatabaseCruncher):
    """
    Database cruncher which uses the 'time-dependent ratio' technique.

    This cruncher derives the relationship between two variables by simply assuming
    that the follower timeseries is equal to the lead timeseries multiplied by a
    time-dependent scaling factor. The scaling factor is the ratio of the
    follower variable to the lead variable. If the database contains many such pairs,
    the scaling factor is the geometric mean of the individual ratios.

    Once the relationship is derived, the 'filler' function will infill following:

    .. math::
        E_f(t) = s(t) * E_l(t)

    where :math:`E_f(t)` is emissions of the follower variable and :math:`E_l(t)` is
    emissions of the lead variable.

    :math:`s` is the scaling factor, calculated as the geometric mean of the ratio of
    the follower to the leader in the cruncher in the database.

    .. math::
        s(t) = geo_mean(\\frac{ E_f(t) }{ E_l(t) })

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
        iamdf_follower, data_follower = self._get_iamdf_followers(
            variable_follower, variable_leaders
        )

        data_follower_unit = np.unique(iamdf_follower.data["unit"].values)
        if data_follower_unit.size == 1:
            data_follower_unit = data_follower_unit[0]
        else:
            raise ValueError("There are multiple/no units in follower data")
        data_follower_time_col = iamdf_follower.time_col
        iamdf_leader = self._db.filter(variable=variable_leaders[0])
        data_leader = iamdf_leader.timeseries()
        if iamdf_leader["unit"].nunique() != 1:
            raise ValueError("There are multiple/no units for the leader data.")
        if data_follower.size != data_leader.size:
            error_msg = "The follower and leader data have different sizes"
            raise ValueError(error_msg)

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
            assert lead_var["unit"].nunique() == 1, \
                "There are multiple units for the variable to infill."
            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )
            times_needed = set(in_iamdf.data[in_iamdf.time_col])
            if any(
                [
                    k not in set(iamdf_follower[data_follower_time_col])
                    for k in times_needed
                ]
            ):
                error_msg = "Not all required timepoints are in the data for " \
                            "the lead gas ({})".format(
                    variable_leaders[0]
                )
                raise ValueError(error_msg)
            output_ts = lead_var.timeseries()

            for year in times_needed:
                # Due to the mathematics of the geometric mean, it is not necessary
                # to ensure the the follower and leader scenarios line up.
                scaling = silicone.stats.geometric_mean(
                    data_follower[year].values / data_leader[year].values
                )
                output_ts[year] = output_ts[year] * scaling
            output_ts.reset_index(inplace=True)
            output_ts["variable"] = variable_follower
            output_ts["unit"] = data_follower_unit

            return IamDataFrame(output_ts)

        return filler

    def _get_iamdf_followers(self, variable_follower, variable_leaders):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherTimeDepRatio`, ``variable_leaders`` should only "
                "contain one variable"
            )

        self._check_follower_and_leader_in_db(variable_follower, variable_leaders)

        iamdf_follower = self._db.filter(variable=variable_follower)
        data_follower = iamdf_follower.timeseries()

        return iamdf_follower, data_follower
