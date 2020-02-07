"""
A wrapper for the 'time-dependent ratio' database cruncher designed for breaking a
composite gas mix into its constituents.
"""

import pandas as pd
from silicone.utils import convert_units_to_MtCO2_equiv

from silicone.database_crunchers import DatabaseCruncherTimeDepRatio


class DecomposeCollectionTimeDepRatio:
    """
    Constructs an aggregate variable and uses the 'time-dependent ratio' technique to
    calculate what this predicts for our database.
    """

    def __init__(self, db):
        """
        Initialises the database to use for infilling.

        Parameters
        ----------
        db : IamDataFrame
            The database for infilling.
        """
        self._db = db.copy()

    def _construct_consistent_values(self, aggregate_name, components, db_to_generate):
        """
            Calculates the sum of the components and creates an IamDataFrame with this
            value under variable type `aggregate_name`.

            Parameters
            ----------
            aggregate_name : str
                The name of the aggregate variable.

            components : [str]
                List of the names of the variables to be summed.

            db_to_generate : :obj:`pyam.IamDataFrame`
                Input data from which to construct consistent values.

            Return
            ------
            :obj:`pyam.IamDataFrame`
                Consistently calculated aggregate data.
            """
        assert (
            aggregate_name not in db_to_generate.variables().values
        ), "We already have a variable of this name"
        relevant_db = db_to_generate.filter(variable=components)
        units = relevant_db.data["unit"].drop_duplicates()
        if len(units) == 0:
            raise ValueError(
                "Attempting to construct a consistent {} but none of the components "
                "present".format(aggregate_name)
            )
        elif len(units) > 1:
            raise ValueError(
                "Too many units found to make a consistent {}".format(aggregate_name)
            )
        combinations = relevant_db.data[
            ["model", "scenario", "region"]
        ].drop_duplicates()
        append_db = []
        for ind in range(len(combinations)):
            model, scenario, region = combinations.iloc[ind]
            case_df = relevant_db.filter(model=model, scenario=scenario, region=region)
            data_to_add = case_df.data.groupby(case_df.time_col).agg("sum")
            for data in data_to_add.iterrows():
                append_db.append(
                    {
                        "model": model,
                        "scenario": scenario,
                        "region": region,
                        "variable": aggregate_name,
                        data_to_add.index.name: data[0],
                        "unit": units.iloc[0],
                        "value": data[1]["value"],
                    }
                )
        return pd.DataFrame(append_db)

    def infill_components(
        self, aggregate, components, to_infill_df, use_ar4_data=False
    ):
        """
        Derive the relationship between the composite variables and their sum, then use
        this to deconstruct the sum.

        Parameters
        ----------
        aggregate : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CO2"``). Unlike in most crunchers, we do not expect the
            database to already contain this data.

        components : list[str]
            The variables whose sum should be equal to the timeseries of the aggregate
             (e.g. ``["Emissions|CO2|AFOLU", "Emissions|CO2|Energy"]``).

        to_infill_df : :obj:`pyam.IamDataFrame`
            The dataframe that already contains the ``aggregate`` variable, but needs
            the ``components`` to be infilled.

        use_ar4_data : bool
            If true, we convert all values to Mt CO2 equivalent using the IPCC AR4
            GWP100 data, otherwise (by default) we use the GWP100 data from AR5.

        Returns
        -------
        :obj:`pyam.IamDataFrame`
            The data for the

        Raises
        ------
        ValueError
            There is no data for ``variable_leaders`` or ``variable_follower`` in the
            database.
        """
        assert (
            aggregate in to_infill_df.variables().values
        ), "The database to infill does not have the aggregate variable"
        assert all(
            y not in components for y in to_infill_df.variables().values
        ), "The database to infill already has some component variables"
        convert_base = convert_units_to_MtCO2_equiv(
            self._db.filter(variable=components), use_AR4_data=use_ar4_data
        )
        db_to_generate = convert_units_to_MtCO2_equiv(
            convert_base, use_AR4_data=use_ar4_data
        )
        consistent_composite = self._construct_consistent_values(
            aggregate, components, db_to_generate
        )
        convert_base.append(consistent_composite, inplace=True)
        cruncher = DatabaseCruncherTimeDepRatio(convert_base)
        df_to_append = []
        for leader in components:
            to_add = cruncher.derive_relationship(leader, [aggregate])(to_infill_df)
            df_to_append.append(to_add)
        return df_to_append
