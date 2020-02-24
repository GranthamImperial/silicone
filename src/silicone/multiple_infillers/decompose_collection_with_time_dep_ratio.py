"""
Uses the 'time-dependent ratio' database cruncher designed for constructing an
aggregate variable and breaking this mix into its constituents.
"""


from silicone.database_crunchers import DatabaseCruncherTimeDepRatio
from silicone.utils import convert_units_to_MtCO2_equiv, _construct_consistent_values


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
        consistent_composite = _construct_consistent_values(
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
