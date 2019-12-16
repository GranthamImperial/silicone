"""
Module for the database cruncher which finds the rolling windows quantile in the data
for a subset of scenarios
"""

import numpy as np
import pandas as pd
import scipy.interpolate
from pyam import IamDataFrame

from .base import _DatabaseCruncher
from ..utils import _get_unit_of_variable
from . import DatabaseCruncherQuantileRollingWindows


class DatabaseCruncherSSPSpecificRelation(_DatabaseCruncher):
    """
    Database cruncher which pre-filters to only use data from specific scenarios.
    Uses the 'rolling windows' technique.

    The nature of the cruncher itself can be found in the 'rolling windows' cruncher
    documentation.
    """

    def derive_relationship(
        self,
        variable_follower,
        variable_leaders,
        quantile=0.5,
        nwindows=10,
        decay_length_factor=1,
        required_scenario="*"
    ):
        """
        Derive the relationship between two variables from the database.

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CH4"``).

        variable_leaders : list[str]
            The variable(s) we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ``["Emissions|CO2"]``).

        quantile : float
            The quantile to return in each window.

        nboxes : int
            The number of windows to use when calculating the relationship between the
            follower and lead gases.

        decay_length_factor : float
            Parameter which controls how strongly points away from the window's centre
            should be weighted compared to points at the centre. Larger values give
            points further away increasingly less weight, smaller values give points
            further away increasingly more weight.

        required_scenario : str or list[str]
            The string which all accepted scenarios are required to match. This may have
            *s to represent wild cards.

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
            There is no data for ``variable_leaders`` or ``variable_follower`` in the
            database.

        ValueError
            ``quantile`` is not between 0 and 1.

        ValueError
            ``nwindows`` is not equivalent to an integer.

        ValueError
            ``decay_length_factor`` is 0.
        """
        use_db = self._db.filter(scenario=required_scenario)
        if use_db.data.empty:
            raise ValueError("There is no data of the appropriate type in the database."
                             " There may be a typo in the SSP option.")
        rolling_windows_cruncher = DatabaseCruncherQuantileRollingWindows(use_db)
        return rolling_windows_cruncher.derive_relationship(
            variable_follower,
            variable_leaders,
            quantile=quantile,
            nwindows=nwindows,
            decay_length_factor=decay_length_factor,
        )
