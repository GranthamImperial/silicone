"""
Modules for the database cruncher which uses the 'constant quantile' technique.
We initially consider a case where there is no known data for the follower emission
"""
import warnings

import numpy as np
from pyam import IamDataFrame
import silicone.utils as utils

from .base import _DatabaseCruncher


class DatabaseCruncherFixedQuantile(_DatabaseCruncher):
    """
    This cruncher takes the first database, with known pathways for some emissions and calculates which quantiles they fit into
    in a second database. It then finds these quantiles of another type of emissions in database two, for which
    there is no data in the first database.
    """
    def _check_data_is_consistent(self, variable_follower, variable_leaders):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherQuantile`, ``variable_leaders`` should only "
                "contain one variable"
            )

        if not all([v in self._db.variables().tolist() for v in variable_leaders]):
            error_msg = "No data for `variable_leaders` ({}) in database".format(
                variable_leaders
            )
            raise ValueError(error_msg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iamdf_follower = self._db.filter(variable=variable_follower)
            iamdf_leader = self._db.filter(variable=variable_leaders)

        if len(variable_follower) < 1:
            raise ValueError(
                "No follower data was requested"
            )
        data_follower = iamdf_follower.data
        if not data_follower.empty:
            error_msg = "There should be no pre-existing data for `variable_follower` ({}) in database".format(
                variable_follower
            )
            raise ValueError(error_msg)


    def derive_relationship(self, variable_follower, variable_leaders):
        self._check_data_is_consistent(variable_follower, variable_leaders)
        variable_follower_db = self._db.filter(variable=variable_follower).pivot_table(
            ["year", "model", "scenario", "region"],
            ["variable"],
            aggfunc="mean"
        )
        variable_leaders_db = self._db.filter(variable=variable_leaders).pivot_table(
                ["year", "model", "scenario", "region"],
                ["variable"],
                aggfunc="mean"
            )
        quantile = utils.which_quantile(variable_leaders_db["year"], variable_leaders["value"])



