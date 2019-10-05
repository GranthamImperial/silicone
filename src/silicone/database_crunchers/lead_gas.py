from .base import _DatabaseCruncher

class DatabaseCruncherLeadGas(_DatabaseCruncher):
    """
    Database cruncher which uses the 'lead gas' technique.

    This cruncher derives the relationship between two variables by simply assuming
    that the follower timeseries is equal to the lead timeseries multiplied by a
    scaling factor. The scaling factor is derived by calculating the ratio of the
    follower variable to the lead variable in the only year in which the follower
    variable is available in the database. As a result, if the follower variable has
    more than one point in the database, this cruncher cannot be used.

    # TODO: turn this into latex which will render properly in the docs

    Mathematically we have:

    E_f(t) = s * E_l(t)

    where E_f(t) is emissions of the follower variable, s is the scaling factor and E_l(t) is emissions of the lead variable.

    s = E_f(t_{fdb}) / E_l(t_{fdb})

    where t_{fdb} is the only time at which the follower gas appears in the database.
    """
    def derive_relationship(self, variable_follower, variable_leaders, **kwargs):
        """
        Derive the relationship between two variables from the database.

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries in future (e.g.
            "C5F12").

        variable_leaders : list[str]
            The variable we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ["CO2"])

        **kwargs
            Keyword arguments used by this class to derive the relationship between
            ``variable_follower`` and ``variable_leaders``.

        Returns
        -------
        :obj:`func`
            Function which takes `pyam.IamDataFrame`s containing `variable_leaders`
            and adds in timeseries for `variable_follower` based on the derived
            relationship between the two.

        Raises
        ------
        ValueError
            ``variable_leaders`` contains more than one variable.

        ValueError
            There is more than one value for ``variable_follower`` in the database.
        """
