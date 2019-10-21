"""
Modules for the database cruncher which uses the 'constant quantile' technique.
We initially consider a case where there is no known data for the follower emission
"""
import warnings
import pandas as pd
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

    def _filter_safely(self, variable_follower, variable_leaders, **kwargs):
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
            error_msg = "No data for `variable_follower` ({}) in database".format(
                variable_follower
            )
            raise ValueError(error_msg)
        if len(iamdf_leader) < 1:
            raise ValueError(
                "No leader data was returned"
            )
        if not len(set(iamdf_follower["unit"])) == 1:
            raise ValueError("Inconsistent units in the original follower data")
        follower_unit = iamdf_follower["unit"].iloc[0]

        pivotted_df_follower = iamdf_follower.pivot_table(index=['year'], columns=['model', 'scenario'], values='value',
                                                          aggfunc='sum')
        pivotted_df_leader = iamdf_leader.pivot_table(index=['year'], columns=['model', 'scenario'], values='value',
                                                      aggfunc='sum')
        return pivotted_df_follower, pivotted_df_leader, follower_unit

    def derive_relationship(self, variable_follower, variable_leaders, **kwargs):
        self._check_data_is_consistent(variable_follower, variable_leaders)

        if "year" not in self._db.data.columns:
            # TODO: replace this with swap_time_for_year function when possible.
            self._db["year"] = pd.DatetimeIndex(self._db["time"]).year

        variable_follower_db, variable_leaders_db, follower_unit = self._filter_safely(
            variable_follower, variable_leaders, **kwargs)
        data_follower_time_col = self._db.time_col
        region = self._db.regions()
        def filler(in_iamdf, interpolate=False):
            """
            Filler function derived from :obj:`DatabaseCruncherFixedQuantile`.

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

            """
            # Check that the data is comparable
            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )
            if all(region != in_iamdf.regions()):
                raise ValueError(
                    "The regions in the filler and the input data are inconsistent"
                )

            in_lead_var = in_iamdf.filter(variable=variable_leaders)
            in_follow_var = in_iamdf.filter(variable=variable_follower)
            if in_lead_var.variables(True).shape[0] != 1:
                raise ValueError("More than one unit detected for input timeseries")
            if "year" not in in_lead_var.data.columns:
                # TODO: replace this with swap_time_for_year function when possible.
                in_lead_var["year"] = pd.DatetimeIndex(in_lead_var["time"]).year
                in_follow_var["year"] = pd.DatetimeIndex(in_follow_var["time"]).year
            in_lead_var = in_lead_var.pivot_table(index=['year'], columns=['model', 'scenario'], values='value',
                       aggfunc='sum')
            output_ts = []

            for column in in_lead_var.columns:
                quantiles_to_match = utils.which_quantile(
                    variable_leaders_db.index.values,
                    variable_leaders_db.values,
                    in_lead_var.index.values,
                    in_lead_var[column].values
                )
                quantile_output = {}
                for year in variable_follower_db.index._data:
                    # TODO: consider a unified definition of quantiles
                    quantile_output.update({year: np.quantile(variable_follower_db.loc[year, ],
                                                                quantiles_to_match.loc[year])})
                dict_to_append = {"model": column[0],
                                  "scenario": column[1],
                                  "region": region[0],
                                  "variable": variable_follower,
                                  "unit": follower_unit}
                dict_to_append.update(quantile_output)
                output_ts.append(dict_to_append)

            return IamDataFrame(pd.DataFrame(output_ts))

        return filler