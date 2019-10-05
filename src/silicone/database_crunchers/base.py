from abc import ABCMeta, abstractmethod, abstractproperty

class _DatabaseCruncher(meta_class=ABCMeta):
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
            The variable for which we want to calculate timeseries in future (e.g.
            "C5F12").

        variable_leaders : list[str]
            The variables we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ["CO2", "CH4"])

        **kwargs
            Keyword arguments used by this class to derive the relationship between
            ``variable_follower`` and ``variable_leaders``.

        Returns
        -------
        :obj:`func`
            Function which takes `pyam.IamDataFrame`s containing `variable_leaders`
            and adds in timeseries for `variable_follower` based on the derived
            relationship between the two.
        """
        # TODO: think about how to add region handling in here...
