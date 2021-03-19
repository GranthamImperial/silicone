"""
Module for the database cruncher which uses the 'latest time quantile' technique.
"""
import logging
import numpy as np
import warnings

from pyam import IamDataFrame
from .base import _DatabaseCruncher
from ..stats import calc_quantiles_of_data

logger = logging.getLogger(__name__)


class ExtendLatestTimeQuantile(_DatabaseCruncher):
    """
    Database cruncher which extends the timeseries of a variable by assuming that it
    remains that a fixed quantile in the infiller database, the quantile it is in at the
    last available time.

    It assumes that the target timeseries is shorter than the infiller timeseries.
    """

    def derive_relationship(self, variable):
        """
        Derives the quantiles of the variable in the infiller database. Note that this
        takes only one variable as an argument, whereas most crunchers take two.

        Parameters
        ----------
        variable : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CO2"``).

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
        data_follower_unit = iamdf.data["unit"]

        assert len(data_follower_unit) == 1, \
            "The infiller database has {} units in it. It should have one. ".format(
                len(data_follower_unit)
            )


        def filler(in_iamdf, interpolate=False):
            """
            Filler function derived from :obj:`LatestTimeRatio`.

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
                    "to generate this filler function (`{}`)".format(
                        infiller_time_col
                    )
                )

            key_timepoint = max(target_df.data[infiller_time_col])
            later_times = [
                t for t in iamdf.data[infiller_time_col] if t > key_timepoint
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
                if filtered.data.empty:
                    if not interpolate:
                        error_msg = (
                            "Required timepoint ({}) is not in the data for "
                            "the variable {}".format(
                                key_timepoint, variable
                            )
                        )
                        raise ValueError(error_msg)
                    idf.interpolate(key_timepoint)
                idf = idf.timeseries()
                if not idf.shape[1] == 1:
                    raise AssertionError(
                        "How did filtering for a single timepoint result in more than "
                        "one column?"
                    )
                return idf.iloc[:, 0]

            infiller_at_key_time = get_values_in_key_timepoint(iamdf)

            target_at_key_time = get_values_in_key_timepoint(target_df)

            quantiles = calc_quantiles_of_data(infiller_at_key_time, target_at_key_time)
            output_ts = target_df.timeseries().reset_index()
            for time in later_times:
                output_ts[time] = np.nanquantile(
                    in_iamdf.filter(**{infiller_time_col: time}).data,
                    quantiles
                )
            return IamDataFrame(output_ts)
        
        return filler

    def _get_iamdf_variable(self, variable):
        if variable not in self._db.variables().tolist():
            error_msg = "No data for `variable` ({}) in database".format(
                variable
            )
            raise ValueError(error_msg)

        return self._db.filter(variable=variable)
