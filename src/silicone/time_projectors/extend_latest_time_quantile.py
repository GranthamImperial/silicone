"""
Module for the database cruncher which uses the 'latest time quantile' technique.
"""
import logging
import warnings

import numpy as np
from pyam import IamDataFrame

from silicone.stats import calc_quantiles_of_data
from silicone.utils import _make_weighting_series

logger = logging.getLogger(__name__)


class ExtendLatestTimeQuantile:
    """
    Time projector which extends the timeseries of a variable by assuming that it
    remains that a fixed quantile in the infiller database, the quantile it is in at the
    last available time. This is the natural counterpart to the equal quantile walk
    extending a single variable over time rather than over different emissions.

    It assumes that the target timeseries is shorter than the infiller timeseries.
    """

    def __init__(self, db):
        """
        Initialise the time projector with a database containing data from the full
        range of times you wish to see in the output.

        Parameters
        ----------
        db : IamDataFrame
            The database to use
        """
        self._db = db.copy()

    def derive_relationship(self, variable, smoothing=None, weighting=None):
        """
        Derives the quantiles of the variable in the infiller database. Note that this
        takes only one variable as an argument, whereas most crunchers take two.

        Parameters
        ----------
        variable : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CO2"``).

        smoothing : float or string
            By default, no smoothing is done on the distribution. If a value is
            provided, it is fed into :func:`scipy.stats.gaussian_kde` - see full
            documentation there. In short, if a float is input, we fit a Gaussian kernel
            density estimator with that width to the points. If a string is used, it
            must be either "scott" or "silverman", after those two methods of
            determining the best kernel bandwidth.

        weighting : None or dict{(str, str): float}
            The dictionary, mapping the (model and scenario) tuple onto the weight (
            relative to a weight of 1 for the default). This does not have to include
            all scenarios in df, but cannot include scenarios not in df.

        Returns
        -------
        :obj:`func`
            Function which takes a :obj:`pyam.IamDataFrame` containing
            ``variable`` timeseries and returns these timeseries extended until the
            latest time in the infiller database.

        Raises
        ------

        ValueError
            There is no data for ``variable`` in the database.

        """
        iamdf = self._get_iamdf_variable(variable)

        infiller_time_col = iamdf.time_col
        data_follower_unit = iamdf.data["unit"].unique()

        assert (
            len(data_follower_unit) == 1
        ), "The infiller database has {} units in it. It should have one. ".format(
            len(data_follower_unit)
        )
        if not isinstance(weighting, type(None)):
            if type(weighting) == dict:
                weighting = _make_weighting_series(iamdf.timeseries(), weighting)
            else:
                raise ValueError("We can only use dictionary values for weights")

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`LatestTimeRatio`.

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
                "The infiller database does not extend in time past the target "
                "database, so no infilling can occur."
            """
            target_df = in_iamdf.filter(variable=variable)
            if target_df.empty:
                error_msg = "No data for `variable` ({}) in target database".format(
                    variable
                )
                raise ValueError(error_msg)
            if infiller_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(infiller_time_col)
                )

            key_timepoint = max(target_df.data[infiller_time_col])
            later_times = [
                t for t in iamdf.data[infiller_time_col].unique() if t > key_timepoint
            ]
            if not later_times:
                raise ValueError(
                    "The infiller database does not extend in time past the target "
                    "database, so no infilling can occur."
                )
            key_timepoint_filter = {infiller_time_col: key_timepoint}

            def get_values_in_key_timepoint(idf):
                # filter warning about empty data frame as we handle it ourselves
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    filtered = idf.filter(**key_timepoint_filter)
                idf = filtered.timeseries()
                if not idf.shape[1] == 1:
                    raise AssertionError(
                        "How did filtering for a single timepoint result in more than "
                        "one column?"
                    )
                return idf.iloc[:, 0]

            infiller_at_key_time = get_values_in_key_timepoint(iamdf)

            target_at_key_time = get_values_in_key_timepoint(target_df)

            quantiles = calc_quantiles_of_data(
                infiller_at_key_time, target_at_key_time, smoothing, weighting
            )
            if any(np.isnan(quantiles)):
                logger.warning("Only a single value provided for calculating quantiles")
                quantiles = [0.5 if np.isnan(q) else q for q in quantiles]
            output_ts = target_df.timeseries()
            iamdf_ts = iamdf.timeseries()
            for time in later_times:
                output_ts[time] = calc_quantiles_of_data(
                    iamdf_ts[time], quantiles, smoothing, weighting, to_quantile=False
                )
            for col in output_ts.columns:
                if col not in later_times:
                    del output_ts[col]
            return IamDataFrame(output_ts)

        return filler

    def _get_iamdf_variable(self, variable):
        if variable not in self._db.variable:
            error_msg = "No data for `variable` ({}) in database".format(variable)
            raise ValueError(error_msg)

        return self._db.filter(variable=variable)
