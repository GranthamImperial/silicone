"""
Uses the 'time-dependent ratio' database cruncher designed for constructing an
aggregate variable and breaking this mix into its constituents.
"""

import pyam

from silicone.database_crunchers import TimeDepRatio
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
            aggregate_name not in db_to_generate.variable
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
        # Units are sorted in alphabetical order so we choose the first to get -equiv
        use["unit"] = units.iloc[0]
        use["variable"] = aggregate_name
        for col in relevant_db.extra_cols:
            use[col] = ""
        return pyam.IamDataFrame(use)

    def _set_of_units_without_equiv(self, df):
        """
        Parameters
        ----------
        df : obj:`pyam.IamDataFrame`
            The dataframe whose units we want

        Returns
        -------
        Set(str)
            The set of units from the dataframe with "-equiv" removed
        """
        return set(df.data["unit"].map(lambda x: x.replace("-equiv", "")))

    def infill_components(
        self,
        aggregate,
        components,
        to_infill_df,
        metric_name="AR5GWP100",
        only_consistent_cases=True,
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

        metric_name : str
            The name of the conversion metric to use. This will usually be
            AR<4/5/6>GWP100.

        only_consistent_cases : bool
            Do we want to only use model/scenario combinations where all aggregate and
            components have data at all times? This will reduce the risk of
            inconsistencies or unevenness in the results, but may reduce the amount of
            data.

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
            aggregate in to_infill_df.variable
        ), "The database to infill does not have the aggregate variable"
        assert all(
            y not in components for y in to_infill_df.variable
        ), "The database to infill already has some component variables"
        assert len(to_infill_df.data.columns) == len(self._db.data.columns) and all(
            to_infill_df.data.columns == self._db.data.columns
        ), (
            "The database and to_infill_db fed into this have inconsistent columns, "
            "which will prevent adding the data together properly."
        )
        self._filtered_db = self._db.filter(variable=components)
        if self._filtered_db.empty:
            raise ValueError(
                "Attempting to construct a consistent {} but none of the components "
                "present".format(aggregate)
            )
        if only_consistent_cases:
            # Remove cases with nans at some time.
            consistent_cases = (
                self._filtered_db.filter(
                    **{
                        to_infill_df.time_col: to_infill_df[
                            to_infill_df.time_col
                        ].unique()
                    }
                )
                .timeseries()
                .dropna()
            )
            self._filtered_db = pyam.IamDataFrame(consistent_cases)

        # We only want to reference cases where all the required components are found
        combinations = self._filtered_db.data[
            ["model", "scenario", "region"]
        ].drop_duplicates()
        for ind in range(len(combinations)):
            model, scenario, region = combinations.iloc[ind]
            found_vars = self._filtered_db.filter(
                model=model, scenario=scenario, region=region
            ).variable
            if any(comp not in found_vars for comp in components):
                self._filtered_db.filter(
                    model=model, scenario=scenario, keep=False, inplace=True
                )
        if len(self._set_of_units_without_equiv(self._filtered_db)) > 1:
            db_to_generate = convert_units_to_MtCO2_equiv(
                self._filtered_db, metric_name=metric_name
            )
        else:
            db_to_generate = self._filtered_db
        consistent_composite = self._construct_consistent_values(
            aggregate, components, db_to_generate
        )
        self._filtered_db.append(consistent_composite, inplace=True)
        cruncher = TimeDepRatio(self._filtered_db)
        if self._set_of_units_without_equiv(
            to_infill_df
        ) != self._set_of_units_without_equiv(consistent_composite):
            raise ValueError(
                "The units of the aggregate variable are inconsistent between the "
                "input and constructed data. We input {} and constructed {}.".format(
                    self._set_of_units_without_equiv(to_infill_df),
                    self._set_of_units_without_equiv(consistent_composite),
                )
            )
        for leader in components:
            to_add = cruncher.derive_relationship(
                leader, [aggregate], only_consistent_cases=False
            )(to_infill_df)
            try:
                df_to_append.append(to_add, inplace=True)
            except NameError:
                df_to_append = to_add
        return df_to_append
