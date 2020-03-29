"""
Module for the database cruncher which uses the 'time-dependent ratio' technique.
"""

import warnings

import numpy as np
import pandas as pd
from pyam import IamDataFrame

from .base import _DatabaseCruncher


class DatabaseCruncherTimeDepRatio(_DatabaseCruncher):
    """
    Database cruncher which uses the 'time-dependent ratio' technique.

    This cruncher derives the relationship between two variables by simply assuming
    that the follower timeseries is equal to the lead timeseries multiplied by a
    time-dependent scaling factor. The scaling factor is the ratio of the
    follower variable to the lead variable. If the database contains many such pairs,
    the scaling factor is the ratio between the means of the values. By default, the
    calculation will include only values where the lead variable takes the same sign
    (+ or -) in the infilling database as in the case infilled. This prevents getting
    negative values of emissions that cannot be negative. To allow cases where we
    have no data of the correct sign, set `same_sign = False` in `derive_relationship`.

    Once the relationship is derived, the 'filler' function will infill following:

    .. math::
        E_f(t) = R(t) * E_l(t)

    where :math:`E_f(t)` is emissions of the follower variable and :math:`E_l(t)` is
    emissions of the lead variable.

    :math:`R(t)` is the scaling factor, calculated as the ratio of the means of the
    the follower and the leader in the infiller database, denoted with
    lower case e. By default, we include only cases where `sign(e_l(t))` is the same in
    both databases).

    .. math::
        R(t) = \\frac{mean( e_f(t) )}{mean( e_l(t) )})

    """

    def derive_relationship(self, variable_follower, variable_leaders, same_sign=True):
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

        same_sign : bool
            Do we want to only use data where the leader has the same sign in the
            infiller and infillee data? If so, we have a potential error from
            not having data of the correct sign, but have more confidence in the
            sign of the follower data.

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
        # Calculate the ratios to use
        all_times = np.unique(iamdf_leader.data[iamdf_leader.time_col])
        scaling = pd.DataFrame(index=all_times, columns=["pos", "neg"])
        if same_sign:
            # We want to have separate positive and negative answers. We calculate a
            # tuple, first for positive and then negative values.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for year in all_times:
                    pos_inds = data_leader[year].values > 0
                    scaling["pos"][year] = np.nanmean(
                        data_follower[year].iloc[pos_inds].values
                    ) / np.nanmean(data_leader[year].iloc[pos_inds].values)
                    scaling["neg"][year] = np.nanmean(
                        data_follower[year].iloc[~pos_inds].values
                    ) / np.nanmean(data_leader[year].iloc[~pos_inds].values)
        else:
            # The tuple is the same in both cases
            for year in all_times:
                scaling["pos"][year] = np.mean(data_follower[year].values) / np.mean(
                    data_leader[year].values
                )
            scaling["neg"] = scaling["pos"]

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`DatabaseCruncherTimeDepRatio`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled-in data (without original source data)

            Raises
            ------
            ValueError
                The key year for filling is not in ``in_iamdf``.
            """
            lead_var = in_iamdf.filter(variable=variable_leaders)
            assert (
                lead_var["unit"].nunique() == 1
            ), "There are multiple units for the lead variable."
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
                error_msg = (
                    "Not all required timepoints are in the data for "
                    "the lead gas ({})".format(variable_leaders[0])
                )
                raise ValueError(error_msg)
            output_ts = lead_var.timeseries()

            for year in times_needed:
                if (
                    scaling.loc[year][
                        output_ts[year].map(lambda x: "neg" if x < 0 else "pos")
                    ]
                    .isnull()
                    .values.any()
                ):
                    raise ValueError(
                        "Attempt to infill {} data using the time_dep_ratio cruncher "
                        "where the infillee data has a sign not seen in the infiller "
                        "database for year "
                        "{}.".format(variable_leaders, year)
                    )
                output_ts[year] = (
                    output_ts[year].values
                    * scaling.loc[year][
                        output_ts[year].map(lambda x: "pos" if x > 0 else "neg")
                    ].values
                )
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
