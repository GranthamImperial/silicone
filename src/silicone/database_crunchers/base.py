from abc import ABCMeta, abstractmethod


class _DatabaseCruncher(metaclass=ABCMeta):
    """
    Base class for database crunching.

    Common operations are shared here, allowing subclasses to just focus on
    implementing the crunching algorithms.
    """

    def __init__(self, db):
        """
        Initialise the database cruncher

        Parameters
        ----------
        db : IamDataFrame
            The database to use
        """
        self._db = db.copy()

    @abstractmethod
    def derive_relationship(self, variable_follower, variable_leaders, **kwargs):
        """
        Derive the relationship between two variables from the database

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|C5F12"``).

        variable_leaders : list[str]
            The variables we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ``["Emissions|CO2", "Emissions|CH4"]``)

        **kwargs
            Keyword arguments used by this class to derive the relationship between
            ``variable_follower`` and ``variable_leaders``.

        Returns
        -------
        :obj:`func`
            Function which takes a :obj:`pyam.IamDataFrame` containing
            ``variable_leaders`` timeseries and returns timeseries for
            ``variable_follower`` based on the derived relationship between the two.
            Please see the source code for the exact definition (and docstring) of the
            returned function.
        """
        # TODO: think about how to add region handling in here...

    def _check_follower_and_leader_in_db(self, variable_follower, variable_leaders):

        if not all([v in self._db.variable for v in variable_leaders]):
            error_msg = "No data for `variable_leaders` ({}) in database".format(
                variable_leaders
            )
            raise ValueError(error_msg)

        if variable_follower not in self._db.variable:
            error_msg = "No data for `variable_follower` ({}) in database".format(
                variable_follower
            )
            raise ValueError(error_msg)
