"""
Module for the database cruncher which uses the 'rolling windows' technique with
different quantiles in different years.
"""
from .base import _DatabaseCruncher
from . import QuantileRollingWindows


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
        nwindows=10,
        decay_length_factor=1,
        use_ratio=False,
    ):
        """
        For details of most parameters, see QuantileRollingWindows. The one different
        parameter is as follows:
        Parameters
        ----------
        time_quantile_dict : dict{datetime or int: float}
            Every year or datetime in the infillee database must be specified as a key.
            The value is the quantile to use in that year. Note that the impact of the
            quantile value is strongly dependent on the choice of nwindows and
            decay_length_factor. This replaces quantile.
        """
        times_known = self.db[self.db.time_col].unique()
        # This check implicitly checks for date type agreement
        if any(time not in times_known for time in time_quantile_dict.keys()):
            raise ValueError(
                "Not all required times in the dictionary have data in the database."
            )
        filler_fns = []
        for time, quantile in time_quantile_dict.items():
            if self.db.time_col == "year":
                cruncher = QuantileRollingWindows(self.db.filter(year=time))
            else:
                cruncher = QuantileRollingWindows(self.db.filter(time=time))
            filler_fns.append(
                cruncher.derive_relationship(
                    variable_follower,
                    variable_leaders,
                    quantile,
                    nwindows,
                    decay_length_factor,
                    use_ratio,
                )
            )

        def filler(in_iamdf):
            iamdf_times_known = in_iamdf[in_iamdf.time_col]
            if any(time not in time_quantile_dict.keys() for time in iamdf_times_known):
                raise ValueError(
                    "Not all required times in the infillee database can be found in"
                    " the dictionary."
                )
            for time in time_quantile_dict.keys():
                if in_iamdf.time_col == "year":
                    filler_fns[0](in_iamdf.filter(year=time))
                else:
                    filler_fns[0](in_iamdf.filter(timw=time))
                filler_fns.pop(0)

        return filler
