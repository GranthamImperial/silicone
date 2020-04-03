from .base import _DatabaseCruncher
from .linear_interpolation import LinearInterpolation


class ScenarioAndModelSpecificInterpolate(_DatabaseCruncher):
    """
    Database cruncher which pre-filters to only use data from specific scenarios, then
    runs the linear interpolator to return values from that set of scenarios. See the
    documentation of LinearInterpolation for more details.
    """

    def derive_relationship(
        self,
        variable_follower,
        variable_leaders,
        required_scenario="*",
        required_model="*",
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

        required_scenario : str or list[str]
            The string(s) which all relevant scenarios are required to match. This may
            have *s to represent wild cards. It defaults to "*" to accept all scenarios.

        required_model : str or list[str]
            The string(s) which all relevant models are required to match. This may have
            *s to represent wild cards. It defaults to "*" to accept all models.

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
             There may be a typo in the SSP option.
        """
        use_db = self._db.filter(scenario=required_scenario, model=required_model)
        if use_db.data.empty:
            raise ValueError(
                "There is no data of the appropriate type in the database."
                " There may be a typo in the SSP option."
            )
        cruncher = LinearInterpolation(use_db)
        return cruncher.derive_relationship(variable_follower, variable_leaders)
