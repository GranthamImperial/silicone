"""
Module for the database cruncher which uses the 'rolling windows' technique with
different quantiles in different years.
"""
from datetime import datetime

import numpy as np

from . import QuantileRollingWindows
from .base import _DatabaseCruncher


class TimeDepQuantileRollingWindows(_DatabaseCruncher):
    """
    Database cruncher which uses QuantileRollingWindows with different quantiles in
    every year/datetime.
    """

    def derive_relationship(
        self,
        variable_follower,
        variable_leaders,
        time_quantile_dict,
        **kwargs,
    ):
        """
        Derive the relationship between two variables from the database.

        For details of most parameters, see QuantileRollingWindows. The one different
        parameter is time_quantile_dict, described below:

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CH4"``).

        variable_leaders : list[str]
            The variable(s) we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ``["Emissions|CO2"]``).

        time_quantile_dict : dict{datetime or int: float}
            Every year or datetime in the infillee database must be specified as a key.
            The value is the quantile to use in that year. Note that the impact of the
            quantile value is strongly dependent on the arguments passed to
            :class:`QuantileRollingWindows`.

        **kwargs
            Passed to :class:`QuantileRollingWindows`.

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
            Not all times in ``time_quantile_dict`` have data in the database.
        """
        if self._db.time_col == "year" and all(
            [isinstance(k, int) for k in time_quantile_dict]
        ):
            time_quantile_dict = {np.int64(k): v for k, v in time_quantile_dict.items()}

        times_known = list(self._db[self._db.time_col].unique())

        # This check implicitly checks for date type agreement
        if any(time not in times_known for time in time_quantile_dict.keys()):
            raise ValueError(
                "Not all required times in the dictionary have data in the database."
            )

        filler_fns = {}
        for time, quantile in time_quantile_dict.items():
            if self._db.time_col == "year":
                cruncher = QuantileRollingWindows(self._db.filter(year=time))
            else:
                cruncher = QuantileRollingWindows(self._db.filter(time=time))
            filler_fns[time] = cruncher.derive_relationship(
                variable_follower, variable_leaders, quantile, **kwargs
            )

        def filler(in_iamdf):
            """
            Filler function derived from :class:`TimeDepQuantileRollingWindows`.

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
                Not all required times in the infillee database have had an
                available interpolation.
            """
            iamdf_times_known = in_iamdf[in_iamdf.time_col]
            if any(
                time not in list(time_quantile_dict.keys())
                for time in iamdf_times_known
            ):
                raise ValueError(
                    "Not all required times in the infillee database can be found in "
                    "the dictionary."
                )

            for time in time_quantile_dict.keys():
                if in_iamdf.time_col == "year":
                    tmp = filler_fns[time](in_iamdf.filter(year=time))
                else:
                    tmp = filler_fns[time](in_iamdf.filter(time=time))

                try:
                    to_return.append(tmp, inplace=True)
                except NameError:
                    to_return = tmp.copy()
            return to_return

        return filler
