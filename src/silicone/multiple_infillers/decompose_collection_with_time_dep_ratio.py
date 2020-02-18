"""
A wrapper for the 'time-dependent ratio' database cruncher designed for breaking a
composite gas mix into its constituents.
"""

import pyam
from silicone.database_crunchers import DatabaseCruncherTimeDepRatio
from silicone.utils import convert_units_to_MtCO2_equiv


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
        units = relevant_db.data["unit"].drop_duplicates().sort_values()
        unit_equivs = units.map(lambda x: x.replace("-equiv", "")).drop_duplicates()
        if len(unit_equivs) == 0:
            raise ValueError(
                "Attempting to construct a consistent {} but none of the components "
                "present".format(aggregate_name)
            )
        elif len(unit_equivs) > 1:
            raise ValueError(
                "Too many units found to make a consistent {}".format(aggregate_name)
            )
        use = (
            relevant_db.data.groupby(
                ["model", "scenario", "region", relevant_db.time_col]
            )
            .agg("sum")
            .reset_index()
        )
        # These are sorted in alphabetical order so we choose the first
        use["unit"] = units.iloc[0]
        use["variable"] = aggregate_name
        return pyam.IamDataFrame(use)

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
            The infilled data resulting from the calculation.

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
        assert len(to_infill_df.data.columns) == len(self._db.data.columns) and all(
            to_infill_df.data.columns == self._db.data.columns
        ), (
            "The database and to_infill_db fed into this have inconsistent columns, "
            "which will prevent adding the data together properly."
        )
        self._db.filter(variable=components, inplace=True)
        # We only want to reference cases where all the required components are found
        combinations = self._db.data[["model", "scenario", "region"]].drop_duplicates()
        for ind in range(len(combinations)):
            model, scenario, region = combinations.iloc[ind]
            found_vars = self._db.filter(
                model=model, scenario=scenario, region=region
            ).variables()
            if any(comp not in found_vars.values for comp in components):
                self._db.filter(
                    model=model, scenario=scenario, keep=False, inplace=True
                )
        convert_base = convert_units_to_MtCO2_equiv(self._db, use_AR4_data=use_ar4_data)
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
            if df_to_append:
                df_to_append.append(to_add)
            else:
                df_to_append = to_add
        return df_to_append
