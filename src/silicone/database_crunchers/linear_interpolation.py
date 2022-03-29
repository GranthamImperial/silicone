"""
Module for the database cruncher which makes a linear interpolator between known values
"""

from warnings import warn

from .interpolation import Interpolation


class LinearInterpolation(Interpolation):
    """
    Database cruncher which uses linear interpolation. This cruncher is deprecated; use
    Interpolation instead.

    This cruncher derives the relationship between two variables by linearly
    interpolating between values in the cruncher database. It does not do any
    smoothing and is best-suited for smaller databases.

    In the case where there is more than one value of the follower variable for a
    given value of the leader variable, the average will be used. For example, if
    one scenario has CH4 emissions of 10 MtCH4/yr whilst another has CH4
    emissions of 20 MtCH4/yr in 2020 whilst both scenarios have CO2 emissions
    of exactly 15 GtC/yr in 2020, the interpolation will use the average value from the
    two scenarios i.e. 15 Mt CH4/yr.

    Beyond the bounds of input data, the interpolation is held constant.
    For example, if the maximum CO2 emissions in 2020 in the database is
    25 GtC/yr, and CH4 emissions for this level of CO2 emissions are 15 MtCH4/yr,
    then even if we infill using a CO2 emissions value of 100 GtC/yr in 2020, the
    returned CH4 emissions will be 15 MtCH4/yr.
    """

    def __init__(self, db):
        warn(
            'This cruncher deprecated, please switch to the more generic interpolation cruncher, "Interpolation"',
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(db)

    def derive_relationship(
        self,
        variable_follower,
        variable_leaders,
    ):
        """
        Derive the relationship between two variables from the database.

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CH4"``).

        variable_leaders : list[str]
            The variable(s) we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ``["Emissions|CO2"]``).

        interpkind : str
            The style of interpolation. By default, linear (hence the name), but can
            also be any value accepted as the "kind" option in
            scipy.interpolate.interp1d.

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
            There is no data of the appropriate type in the database.
        """
        return super().derive_relationship(
            variable_follower=variable_follower,
            variable_leaders=variable_leaders,
            interpkind="linear",
        )
