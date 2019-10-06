from pyam import IamDataFrame

from .base import _DatabaseCruncher


class DatabaseCruncherLeadGas(_DatabaseCruncher):
    """
    Database cruncher which uses the 'lead gas' technique.

    This cruncher derives the relationship between two variables by simply assuming
    that the follower timeseries is equal to the lead timeseries multiplied by a
    scaling factor. The scaling factor is derived by calculating the ratio of the
    follower variable to the lead variable in the only year in which the follower
    variable is available in the database. As a result, if the follower variable has
    more than one point in the database, this cruncher cannot be used. Additionally,
    the derived relationship only depends on a single point in the database, no
    regressions or other calculations are performed.

    # TODO: turn this into latex which will render properly in the docs

    Mathematically we have:

    E_f(t) = s * E_l(t)

    where E_f(t) is emissions of the follower variable, s is the scaling factor and E_l(t) is emissions of the lead variable.

    s = E_f(t_{fdb}) / E_l(t_{fdb})

    where t_{fdb} is the only time at which the follower gas appears in the database.
    """

    def derive_relationship(self, variable_follower, variable_leaders, **kwargs):
        """
        Derive the relationship between two variables from the database.

        Parameters
        ----------
        variable_follower : str
            The variable for which we want to calculate timeseries in future (e.g.
            "C5F12").

        variable_leaders : list[str]
            The variable we want to use in order to infer timeseries of
            ``variable_follower`` (e.g. ["CO2"]). Note that the 'lead gas' methodology
            gives the same result, indepent of the value of `variable_leaders` in the
            database.

        **kwargs
            Keyword arguments used by this class to derive the relationship between
            ``variable_follower`` and ``variable_leaders``.

        Returns
        -------
        :obj:`func`
            Function which takes `pyam.IamDataFrame`s containing `variable_leaders`
            timeseries and returns timeseries for `variable_follower` based on the
            derived relationship between the two.

        Raises
        ------
        ValueError
            ``variable_leaders`` contains more than one variable.

        ValueError
            There is more than one value for ``variable_follower`` in the database.
        """
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherLeadGas`, ``variable_leaders`` should only contain one variable"
            )

        if not all([v in self._db.variables().tolist() for v in variable_leaders]):
            error_msg = "No data for `variable_leaders` ({}) in database".format(
                variable_leaders
            )
            raise ValueError(error_msg)

        iamdf_follower = self._db.filter(variable=variable_follower)
        data_follower = iamdf_follower.data
        if data_follower.shape[0] != 1:
            raise ValueError

        data_follower_time_col = iamdf_follower.time_col
        data_follower_key_timepoint = data_follower[
            data_follower_time_col
        ].values.squeeze()
        data_follower_key_year_val = data_follower["value"].values.squeeze()
        data_follower_unit = data_follower["unit"].values[0]

        def filler(in_iamdf, interpolate=False):
            """
            Filler function derived from :obj:`DatabaseCruncherLeadGas`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            interpolate : bool
                If the key year for filling is not in ``in_iamdf``, should a value be
                interpolated?

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled in data (without original source data)

            Raises
            ------
            ValueError
                The key year for filling is not in ``in_iamdf`` and ``interpolate is
                False``.
            """
            import pdb

            pdb.set_trace()
            # unit check (can be conversion in future)
            lead_var = in_iamdf.filter(variable=variable_leaders)

            # for other crunchers, unit check would look like this (doesn't actually
            # matter for this cruncher)
            import pdb

            pdb.set_trace()
            var_units = lead_var.variables(True)
            if var_units.shape[0] != 1:
                raise ValueError("More than one unit detected for input timeseries")
            if (
                var_units.set_index("variable")[variable_leaders[0]]["unit"]
                != "expected_unit"
            ):
                raise ValueError(
                    "Units of lead variable is meant to be `expected_unit`, found `other_unit`"
                )

            key_timepoint_filter = {data_follower_time_col: data_follower_key_timepoint}
            lead_var_val_in_key_timepoint = lead_var.filter(
                **key_timepoint_filter
            ).timeseries()
            scaling = data_follower_key_year_val / lead_var_val_in_key_timepoint
            output_ts = (lead_var.timeseries() * scaling).reset_index()
            output_ts["variable"] = variable_follower
            output_ts["unit"] = data_follower_unit

            return IamDataFrame(output_ts)

        return filler
