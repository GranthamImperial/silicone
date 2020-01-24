"""
A wrapper for the 'time-dependent ratio' database cruncher designed for breaking a
composite gas mix into its constituents.
"""

import numpy as np
import pandas as pd
from pyam import IamDataFrame

from .base import _DatabaseCruncher
from silicone.database_crunchers import DatabaseCruncherTimeDepRatio


class DatabaseCruncherGasDecomposeTimeDepRatio(_DatabaseCruncher):
    """
    Database cruncher which uses the 'time-dependent ratio' technique.
    """

    def _construct_consistent_values(self, aggregate_name, component_ratio):
        """
            Calculates the sum of the components with a given ratio and adds the value
            to the database in self.

            Parameters
            ----------
            aggregate_name : Str
                The name of the aggregate variable

            component_ratio : [Str]
                List of the names of the variables to be summed

            Return
            ------
            (Entry is added directly to self._db)
            """
        assert aggregate_name not in self._db.variables(), "We already have a " \
                                                           "variable of this name"
        relevant_db = self._db.filter(variable=component_ratio)
        units = relevant_db.data["unit"].drop_duplicates()
        if len(units) == 0:
            print("Attempting to construct a consistent {} but none of the components "
                  "present".format(aggregate_name))
            return
        elif len(units) > 1:
            raise ValueError("Too many units found to make a consistent {}".format(aggregate_name))
        combinations = relevant_db.data[["model", "scenario", "region"]].drop_duplicates()
        append_db = []
        for ind in range(len(combinations)):
            model, scenario, region = combinations.iloc[ind]
            case_df = relevant_db.filter(
                model=model, scenario=scenario, region=region
            )
            if case_df.data.empty:
                continue
            data_to_add = case_df.data.groupby(case_df.time_col).agg("sum")
            for data in data_to_add.iterrows():
                append_db.append({
                    "model": model, "scenario": scenario, "region": region, "variable": aggregate_name, data_to_add.index.name: data[0], "unit": units[0], "value": data[1]
                })
        self._db.append(pd.DataFrame(append_db), inplace=True)

    def derive_relationship(self, variable_follower, variable_leaders):
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
            ``variable_leaders`` contains more than one variable.

        ValueError
            There is no data for ``variable_leaders`` or ``variable_follower`` in the
            database.
        """
        use_db = self._db
        cruncher = DatabaseCruncherTimeDepRatio(use_db)
        return cruncher.derive_relationship(variable_follower, variable_leaders)
