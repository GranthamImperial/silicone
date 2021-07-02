"""
Module for the database cruncher which extends using a linear trend
"""
import datetime
import logging
import warnings

from pyam import IamDataFrame

logger = logging.getLogger(__name__)


class LinearExtender:
    """
    Time projector which extends the timeseries of a variable using a linear trend. You
    can either specify a gradient for the line (possibly zero) or a point in the future.
    """

    def __init__(self, db=None):
        """
        Initialise the time projector. The database is only used to determine the
        required timeperiods to return and can be left out if this is specified later.

        Parameters
        ----------
        db : IamDataFrame or None
            The database to use to determine times to return
        """
        if db:
            self._db = db.copy()
        else:
            self._db = db

    def derive_relationship(self, variable, gradient=None, year_value=None, times=None):
        """
        Derives the function to return a linear trend following from the last datapoint

        Parameters
        ----------
        variable : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CO2"``).

        gradient : float or None
            The gradient of the variable after its last available datapoint, in the
            emissions units per year. If not provided, year_value must be provided
            instead.

        year_value : None or tuple(int or datetime, float)
            The value of the variable at a given future time, e.g. (2050, 0) to extend
            the data to net zero in 2050. If not provided, gradient must be provided
            instead.

        times : None or list[int or datetime]
            The times to return entries at. Only required if no database was used during
            initalisation.


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
        if times is None:
            if not self._db:
                raise ValueError(
                    "This function must either be given a list of times or a "
                    "database of completed scenarios"
                )
            times = self._db[self._db.time_col].unique()

        if (gradient is not None) and (year_value is not None):
            raise ValueError("Provide only one of a year_value OR gradient")

        if (gradient is None) and (year_value is None):
            raise ValueError("Provide either a year_value OR gradient")

        if year_value:
            if (not isinstance(year_value, tuple)) or (len(year_value) != 2):
                raise ValueError(
                    "year_value should be a tuple of the year and the value that year."
                )

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`LinearExtender`.

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
            infiller_time_col = target_df.time_col
            last_time = max(target_df.data[infiller_time_col])
            try:
                later_times = [time for time in times if time > last_time]
            except TypeError as exc:
                raise TypeError(
                    "The times requested must be in the same format as the time column "
                    "in the input database"
                ) from exc

            if not later_times:
                raise ValueError(
                    "No times requested are later than the times already in "
                    "the database"
                )
            key_timepoint_filter = {infiller_time_col: last_time}

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

            target_at_key_time = get_values_in_key_timepoint(target_df)

            output_ts = target_df.timeseries()
            # one_year required so type conversion works properly
            one_year = (
                1 if (infiller_time_col == "year") else datetime.timedelta(days=365)
            )
            if year_value:
                gradient_to_use = (year_value[1] - target_at_key_time) / (
                    (year_value[0] - last_time) / one_year
                )
            else:
                gradient_to_use = gradient

            for time in later_times:
                output_ts[time] = target_at_key_time + gradient_to_use * (
                    (time - last_time) / one_year
                )

            for col in output_ts.columns:
                if col not in later_times:
                    del output_ts[col]

            return IamDataFrame(output_ts)

        return filler
