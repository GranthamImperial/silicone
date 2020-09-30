"""
Module for the database cruncher which uses the 'constant given ratio' technique.
"""
import logging

import pandas as pd
from pyam import IamDataFrame

from .base import _DatabaseCruncher

logger = logging.getLogger(__name__)


class ConstantRatio(_DatabaseCruncher):
    """
    Database cruncher which uses the 'constant given ratio' technique.

    This cruncher does not require a database upon initialisation. Instead, it requires
    a constant and a unit to be input when deriving relations. This constant,
    :math:`s`, is the ratio of the follower variable to the lead variable i.e.:

    .. math::

        E_f(t) = s * E_l(t)

    where :math:`E_f(t)` is emissions of the follower variable and :math:`E_l(t)` is
    emissions of the lead variable.
    """

    def __init__(self, db=None):
        """
        Initialise the database cruncher

        Parameters
        ----------
        db : IamDataFrame
            Supplied to ensure consistency with the base class. This cruncher doesn't
            actually use the database at all.
        """
        if db is not None:
            logger.info(
                "%s won't use any information from the database", self.__class__
            )

        super().__init__(pd.DataFrame())

    def derive_relationship(self, variable_follower, variable_leaders, ratio, units):
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

        ratio : float
            The ratio between the leader and the follower data

        units : str
            The units of the follower data.

        Returns
        -------
        :obj:`func`
            Function which takes a :obj:`pyam.IamDataFrame` containing
            ``variable_leaders`` timeseries and returns timeseries for
            ``variable_follower`` based on the derived relationship between the two.
        """
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `ConstantRatio`, ``variable_leaders`` should only "
                "contain one variable"
            )

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`TimeDepRatio`.

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
            output_ts = in_iamdf.filter(variable=variable_leaders).data
            if any(output_ts["value"] < 0):
                warn_str = "Note that the lead variable {} goes negative.".format(
                    variable_leaders
                )
                logger.warning(warn_str)
                print(warn_str)
            assert (
                output_ts["unit"].nunique() == 1
            ), "There are multiple or no units for the lead variable."
            output_ts["value"] = output_ts["value"] * ratio
            output_ts["variable"] = variable_follower
            output_ts["unit"] = units

            return IamDataFrame(output_ts)

        return filler
