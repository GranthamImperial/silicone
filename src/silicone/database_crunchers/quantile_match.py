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
    def _check_data_is_consistent(self, variable_follower, variable_leaders, **kwargs):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherQuantile`, ``variable_leaders`` should only "
                "contain one variable"
            )

        if len(self._db.regions()) > 1:
            error_msg = "Multiple regions are being treated simultaneously."
            raise ValueError(error_msg)

        if not all([v in self._db.variables().tolist() for v in variable_leaders]):
            error_msg = "No data for `variable_leaders` ({}) in database".format(
                variable_leaders
            )
            raise ValueError(error_msg)

        if not all([variable_follower in self._db.variables().tolist()]):
            error_msg = "No data for `variable_follower` ({}) in database".format(
                variable_follower
            )
            raise ValueError(error_msg)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            iamdf_follower = self._db.filter(variable=variable_follower, **kwargs)
            iamdf_leader = self._db.filter(variable=variable_leaders)

        if len(variable_follower) < 1:
            raise ValueError(
                "No follower data was returned"
            )
        data_follower = iamdf_follower.data
        if data_follower.empty:
            error_msg = "There should be pre-existing data for `variable_follower` ({}) in database".format(
                variable_follower
            )
            raise ValueError(error_msg)
        if len(iamdf_leader) <1:
            raise ValueError(
                "No leader data was returned"
            )


    def derive_relationship(self, variable_follower, variable_leaders, **kwargs):
        self._check_data_is_consistent(variable_follower, variable_leaders, **kwargs)
        variable_follower_db = self._db.filter(variable=variable_follower, **kwargs)\
            .pivot_table(index=['year'], columns=['model', 'scenario'], values='value', aggfunc='sum')
        variable_leaders_db = self._db.filter(variable=variable_leaders)\
            .pivot_table(index=['year'], columns=['model', 'scenario'], values='value', aggfunc='sum')
        quantile_to_match = utils.which_quantile(
            variable_leaders_db[:].index.values,
            variable_leaders_db[:].values,
            variable_follower_db[:].index.values,
            variable_follower_db[:].values
        )


        return quantile_to_match

