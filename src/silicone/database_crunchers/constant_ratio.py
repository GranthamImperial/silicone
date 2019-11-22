"""
Module for the database cruncher which uses the 'constant given ratio' technique.
"""

from pyam import IamDataFrame

from .base import _DatabaseCruncher


class DatabaseCruncherConstantRatio(_DatabaseCruncher):
    """
    Database cruncher which uses the 'constant given ratio' technique.

    This cruncher does not require a database upon initialisation. Instead, it requires
    a constant and a unit to be input when deriving relations. This constant is the
    ratio of the follower variable to the lead variable, :math:`s`, where:
    .. math::
        E_f(t) = s * E_l(t)

    for :math:`E_f(t)` the emissions of the follower variable and :math:`E_l(t)` the
    emissions of the lead variable.
    """

    def __init__(self):
        db = None

    def derive_relationship(self, variable_follower, variable_leaders, ratio, units):
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

        ratio : float
            The ratio between the leader and the follower data

        units : str
            The units of the follower data.

        Returns
        -------
        :obj:`func`
            Function which takes a :obj:`pyam.IamDataFrame` containing
            ``variable_leaders`` timeseries and returns timeseries for
            ``variable_follower`` based on the derived relationship between the two.
        """
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherConstantRatio`, ``variable_leaders`` should only "
                "contain one variable"
            )

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`DatabaseCruncherTimeDepRatio`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled-in data (without original source data)

            Raises
            ------
            ValueError
                The key year for filling is not in ``in_iamdf``.
            """
            lead_var = in_iamdf.filter(variable=variable_leaders)
            assert (
                lead_var["unit"].nunique() == 1
            ), "There are multiple units for the lead variable."
            times_needed = set(in_iamdf.data[in_iamdf.time_col])
            output_ts = lead_var.timeseries()
            for year in times_needed:
                output_ts[year] = output_ts[year] * ratio
            output_ts.reset_index(inplace=True)
            output_ts["variable"] = variable_follower
            output_ts["unit"] = units

            return IamDataFrame(output_ts)

        return filler
