"""
Uses the same cruncher (by default the 'quantile rolling windows') to split an aggregate
emission into all but one of its constituents and infills the remainder as another
specified emissions type (which may be negative).
"""

import logging

from silicone.database_crunchers import QuantileRollingWindows
from silicone.multiple_infillers import infill_composite_values
from silicone.utils import _remove_equivs, convert_units_to_MtCO2_equiv

logger = logging.getLogger(__name__)


class SplitCollectionWithRemainderEmissions:
    """
    Splits the known aggregate emissions into several components with a specified
    cruncher (by default the 'quantile rolling windows' cruncher), then allocates any
    remainder to ``remainder``.
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

    def _check_and_return_desired_unit(
        self, relevant_df, aggregate, components, remainder
    ):
        """
        Converts the units of the component emissions to be the same as the
        aggregate emissions. Returns the converted database and the unit.

        Parameters
        ----------
        relevant_df : :obj:`pyam.IamDataFrame`
            Data with units that need correcting.

        aggregate : str
            The name of the aggregate variable.

        components : [str]
            List of the names of the variables to be summed.

        remainder : str
            The component which will be constructed as a remainder.

        Return
        ------
        :obj:`pyam.IamDataFrame`
            Data with consistent units.

        str
            The unit of the aggregate data.

        Raises
        ------
        ValueError
            The variables in this dataframe have units that cannot easily be converted
            to make them consistent.
        """
        all_var = [aggregate, remainder] + components
        all_units = relevant_df.data[["variable", "unit"]].drop_duplicates()
        if not all(var in all_units["variable"].values for var in all_var):
            logger.warning(
                "Some variables missing from database when performing "
                "unit conversion: {}".format(
                    [var for var in all_var if var not in all_units["variable"]]
                )
            )
        assert (
            aggregate in all_units["variable"].values
        ), "No aggregate data in database."
        if remainder:
            assert (
                remainder in all_units["variable"].values
            ), "No remainder data in database."
        desired_unit = all_units["unit"][all_units["variable"] == aggregate]
        assert len(desired_unit) == 1, "Multiple units for the aggregate variable"
        desired_unit = desired_unit.iloc[0]
        desired_unit_eqiv = _remove_equivs(desired_unit)
        unit_equivs = all_units["unit"].map(_remove_equivs).drop_duplicates()
        if len(unit_equivs) != 1 and desired_unit_eqiv != "Mt CO2/yr":
            raise ValueError(
                "The variables in this dataframe have units that cannot "
                "easily be converted to make them consistent."
            )
        return relevant_df, desired_unit

    def infill_components(
        self,
        aggregate,
        components,
        remainder,
        to_infill_df,
        cruncher_class=QuantileRollingWindows,
        metric_name="AR5GWP100",
        **kwargs,
    ):
        """
        Derive the relationship between the composite variables and their sum, then use
        this to deconstruct the sum.

        Parameters
        ----------
        aggregate : str
            The variable for which we want to calculate timeseries (e.g.
            ``"Emissions|CO2"``).

        components : list[str]
            The variables to be infilled directly by the cruncher. (e.g.
            ``["Emissions|CO2|AFOLU", "Emissions|CO2|Energy"]``).

        remainder : str
            The variable which will absorb any difference between the aggregate and
            component emissions. This may be positive or negative. E.g.
            ``"Emissions|CO2|Industry"``

        to_infill_df : :obj:`pyam.IamDataFrame`
            The dataframe that already contains the ``aggregate`` variable, but needs
            the ``components`` to be infilled.

        cruncher_class : :class:
            The cruncher used to perform the infilling. By default this will be
            QuantileRollingWindows.

        metric_name : str
            The name of the conversion metric to use. This will usually be
            AR<4/5/6>GWP100.

        **kwargs :
            Handed to the cruncher.

        Returns
        -------
        :obj:`pyam.IamDataFrame`
            The infilled data resulting from the calculation.
        """
        assert (
            aggregate in to_infill_df.variable
        ), "The database to infill does not have the aggregate variable"
        assert all(
            y not in [remainder] + components for y in to_infill_df.variable
        ), "The database to infill already has some component variables"
        assert len(to_infill_df.data.columns) == len(self._db.data.columns) and all(
            to_infill_df.data.columns == self._db.data.columns
        ), (
            "The database and to_infill_db fed into this have inconsistent columns, "
            "which will prevent adding the data together properly."
        )
        to_infill_df = to_infill_df.filter(variable=aggregate)
        assert (
            len(to_infill_df["unit"].unique()) == 1
        ), "Multiple units in the aggregate data"
        to_infill_ag_units = to_infill_df.unit[0]
        all_var = [aggregate, remainder] + components
        relevant_df = self._db.filter(variable=all_var)
        db_to_generate, aggregate_unit = self._check_and_return_desired_unit(
            relevant_df, aggregate, components, remainder
        )
        assert _remove_equivs(aggregate_unit) == _remove_equivs(to_infill_ag_units), (
            "The units of the aggregate variable are different between infiller and "
            "infillee dataframes"
        )
        cruncher = cruncher_class(db_to_generate)
        unavailable_comp = [
            var for var in components if var not in relevant_df.variable
        ]
        if unavailable_comp:
            logger.warning("No data found for {}".format(unavailable_comp))
        remainder_dict = {aggregate: 1}
        components = [comp for comp in components if comp not in unavailable_comp]
        for leader in components:
            to_add = cruncher.derive_relationship(leader, [aggregate], **kwargs)(
                to_infill_df
            )
            try:
                df_to_append.append(to_add, inplace=True)
            except NameError:
                df_to_append = to_add
            remainder_dict[leader] = -1
        # We need a single database with both aggregate and components, which we may
        # want to convert units to CO2/yr before we calculate the remainder.
        calculate_remainder_df = df_to_append.append(to_infill_df)
        if _remove_equivs(aggregate_unit) == "Mt CO2/yr":
            calculate_remainder_df = convert_units_to_MtCO2_equiv(
                calculate_remainder_df, metric_name
            )

        df_to_append.append(
            infill_composite_values(
                calculate_remainder_df, {remainder: remainder_dict}
            ),
            inplace=True,
        )
        return df_to_append
