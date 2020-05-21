"""
Uses the 'quantile rolling windows' database cruncher to infill most constituents and
infills the remainder as another specified emissions type (which may be negative).
"""

import logging

from silicone.database_crunchers import QuantileRollingWindows
from silicone.multiple_infillers import infill_composite_values
from silicone.utils import convert_units_to_MtCO2_equiv, _remove_equivs

logger = logging.getLogger(__name__)

class SplitCollectionWithRemainderEmissions:
    """
    Splits the known aggregate emissions into several components with the 'quantile
    rolling windows' cruncher, then sets the remainder equal to the
    'remainder_emissions'.
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

    def _make_units_consistent(
            self, df, aggregate, components, remainder, use_ar4_data
    ):
        """
            Converts the units of the component emissions to be the same as the
            aggregate emissions. Returns the converted database and the unit.

            Parameters
            ----------
            df : :obj:`pyam.IamDataFrame`
                Data with units that need correcting.

            aggregate : str
                The name of the aggregate variable.

            components : [str]
                List of the names of the variables to be summed.

            remainder : str
                The component which will be constructed as a remainder

            Return
            ------
            :obj:`pyam.IamDataFrame`
                Data with consistent units.

            str
                The unit of the aggregate data.
            """
        all_var = [aggregate] + components + [remainder]
        relevant_df = df.filter(variable=all_var)
        all_units = relevant_df.variables(True)
        if not all(var in all_units["variable"].values for var in all_var):
            "Some variables missing from database when performing unit " \
            "conversion: {}".format(
                [var not in all_units["variable"] for var in all_var]
            )
        assert aggregate in all_units["variable"].values, \
            "No aggregate data in database."
        assert remainder in all_units["variable"].values, \
            "No remainder data in database."
        desired_unit = all_units["unit"][all_units["variable"] == aggregate]
        assert len(desired_unit) == 1, "Multiple units for the aggregate variable"
        desired_unit = desired_unit[0]
        desired_unit_eqiv = _remove_equivs(desired_unit)
        unit_equivs = all_units["unit"].map(_remove_equivs).drop_duplicates()
        if len(unit_equivs) == 1:
            return relevant_df, desired_unit
        if desired_unit_eqiv == "Mt CO2/yr":
            return convert_units_to_MtCO2_equiv(relevant_df, use_ar4_data), desired_unit
        else:
            raise ValueError("The variables in this dataframe have units that cannot "
                             "easily be converted to make them consistent.")

    def infill_components(
        self, aggregate, components, remainder, to_infill_df, use_ar4_data=False, **kwargs
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
            The variables to be infilled by quantile rolling window method. (e.g.
            ``["Emissions|CO2|AFOLU", "Emissions|CO2|Energy"]``).The sum of
            these will be equal to the timeseries of the aggregate minus the remainder
             term.

        remainder : str
            The variable which will absorb any difference between the aggregate and
            component emissions. This may be positive or negative. Typically this will
            be ``"Emissions|CO2"``

        to_infill_df : :obj:`pyam.IamDataFrame`
            The dataframe that already contains the ``aggregate`` variable, but needs
            the ``components`` to be infilled.

        use_ar4_data : bool
            If true, we convert all values to Mt CO2 equivalent using the IPCC AR4
            GWP100 data, otherwise (by default) we use the GWP100 data from AR5.

        **kwargs :
            An optional dictionary of instructions handed to the quantile rolling
            windows cruncher.

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
            y not in [remainder] + components for y in to_infill_df.variables().values
        ), "The database to infill already has some component variables"
        assert len(to_infill_df.data.columns) == len(self._db.data.columns) and all(
            to_infill_df.data.columns == self._db.data.columns
        ), (
            "The database and to_infill_db fed into this have inconsistent columns, "
            "which will prevent adding the data together properly."
        )
        to_infill_df = to_infill_df.filter(variable=aggregate)
        to_infill_ag_units = to_infill_df.variables(True)["unit"].values[0]
        db_to_generate, aggregate_unit = self._make_units_consistent(
            self._db, aggregate, components, remainder, use_ar4_data
        )
        assert _remove_equivs(aggregate_unit) == _remove_equivs(to_infill_ag_units), \
            "The units of the aggregate variable are different between infiller and " \
            "infillee dataframes"
        cruncher = QuantileRollingWindows(db_to_generate)
        for leader in components:
            to_add = cruncher.derive_relationship(leader, [aggregate], **kwargs)(
                to_infill_df
            )
            try:
                df_to_append.append(to_add, inplace=True)
            except NameError:
                df_to_append = to_add
        calculate_remainder_df = df_to_append.append(to_infill_df)
        remainder_dict = {aggregate: 1}
        for item in components:
            remainder_dict[item] = -1
        df_to_append.append(
            infill_composite_values(
                calculate_remainder_df, {remainder: remainder_dict}
            ),
            inplace=True
        )
        return df_to_append
