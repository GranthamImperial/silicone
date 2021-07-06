"""
Module for the database cruncher which uses the 'equal quantile walk' technique.
"""

import numpy as np
from pyam import IamDataFrame

from ..stats import calc_quantiles_of_data
from ..utils import _make_weighting_series
from .base import _DatabaseCruncher


class EqualQuantileWalk(_DatabaseCruncher):
    """
    Database cruncher which uses the 'equal quantile walk' technique.

    This cruncher assumes that the amount of effort going into reducing one emission set
    is equal to that for another emission, therefore the lead and follow data should be
    at the same quantile of all pathways in the infiller database.
    It calculates the quantile of the lead infillee data in the lead infiller database,
    then outputs that quantile of the follow data in the infiller database.
    """

    def derive_relationship(
        self, variable_follower, variable_leaders, smoothing=None, weighting=None
    ):
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

        smoothing : float or string
            By default, no smoothing is done on the distribution. If a value is
            provided, it is fed into :func:`scipy.stats.gaussian_kde` - see full
            documentation there. In short, if a float is input, we fit a Gaussian kernel
            density estimator with that width to the points. If a string is used, it
            must be either "scott" or "silverman", after those two methods of
            determining the best kernel bandwidth.

        weighting: Dict{(str, str) : float}
            The dictionary, mapping the (mode, scenario) tuple onto the weight (relative
            to a weight of 1 for the default). This does not have to include all scenarios
            in df, but cannot include scenarios not in df.

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
        follower_ts = iamdf_follower.timeseries()

        data_follower_time_col = iamdf_follower.time_col
        data_follower_unit = iamdf_follower["unit"].values[0]
        lead_ts = self._db.filter(variable=variable_leaders).timeseries()
        if weighting is not None:
            if isinstance(weighting, dict):
                weighting_follow = _make_weighting_series(follower_ts, weighting)
                weighting_lead = _make_weighting_series(lead_ts, weighting)
            else:
                raise TypeError("``weighting`` should be a dictionary")
        else:
            weighting_follow = None
            weighting_lead = None

        lead_unit = lead_ts.index.get_level_values("unit")[0]

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`EqualQuantileWalk`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled in data (without original source data)

            Raises
            ------
            ValueError
                Not all required timepoints are present in the database we crunched...
            """
            lead_in = in_iamdf.filter(variable=variable_leaders)
            if not all([unit == lead_unit for unit in lead_in.unit]):
                raise ValueError(
                    "Units of lead variable is meant to be `{}`, found `{}`".format(
                        lead_unit, lead_in.unit
                    )
                )

            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )
            if lead_in.data.empty:
                raise ValueError(
                    "There is no data for {} so it cannot be infilled".format(
                        variable_leaders
                    )
                )
            output_ts = lead_in.timeseries()
            if any(
                [
                    (time not in lead_ts.columns) or (time not in follower_ts.columns)
                    for time in output_ts.columns
                ]
            ):
                # We allow for cases where either lead or follow have gaps
                raise ValueError(
                    "Not all required timepoints are present in the database we "
                    "crunched, we crunched \n\t{} for the lead and \n\t{} for the "
                    "follow \nbut you passed in \n\t{}".format(
                        lead_ts.columns, follower_ts.columns, output_ts.columns
                    )
                )
            for col in output_ts.columns:
                output_ts[col] = self._find_same_quantile(
                    follower_ts[col],
                    lead_ts[col],
                    output_ts[col],
                    smoothing,
                    weighting_lead,
                    weighting_follow,
                )
            output_ts = output_ts.reset_index()
            output_ts["variable"] = variable_follower
            output_ts["unit"] = data_follower_unit
            return IamDataFrame(output_ts)

        return filler

    def _get_iamdf_follower(self, variable_follower, variable_leaders):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `EqualQuantileWalk`, ``variable_leaders`` should only "
                "contain one variable"
            )

        self._check_follower_and_leader_in_db(variable_follower, variable_leaders)

        return self._db.filter(variable=variable_follower)

    def _find_same_quantile(
        self,
        follow_vals,
        lead_vals,
        lead_input,
        smoothing,
        weighting_lead,
        weighting_follow,
    ):
        # Dispose of nans that can cloud the calculation
        follow_vals = follow_vals[~np.isnan(follow_vals)]
        input_quantiles = calc_quantiles_of_data(
            lead_vals, lead_input, smoothing, weighting_lead
        )
        if all(np.isnan(input_quantiles)):
            return np.nanmean(follow_vals)
        return calc_quantiles_of_data(
            follow_vals, input_quantiles, smoothing, weighting_follow, to_quantile=False
        )
