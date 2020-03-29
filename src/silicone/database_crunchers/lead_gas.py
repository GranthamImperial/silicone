"""
Module for the database cruncher which uses the 'lead gas' technique.
"""
import warnings

import numpy as np
from pyam import IamDataFrame

from .base import _DatabaseCruncher


class DatabaseCruncherLeadGas(_DatabaseCruncher):
    """
    Database cruncher which uses the 'lead gas' technique.

    This cruncher derives the relationship between two variables by simply assuming
    that the follower timeseries is equal to the lead timeseries multiplied by a
    scaling factor. The scaling factor is derived by calculating the ratio of the
    follower variable to the lead variable in the latest year in which the follower
    variable is available in the database. Additionally, since
    the derived relationship only depends on a single point in the database, no
    regressions or other calculations are performed.

    Once the relationship is derived, the 'filler' function will infill following:

    .. math::
        E_f(t) = R * E_l(t)

    where :math:`E_f(t)` is emissions of the follower variable and :math:`E_l(t)` is
    emissions of the lead variable, both in the infillee database.

    :math:`R` is the scaling factor, calculated as

    .. math::
        R = \\frac{ E_f(t_{\\text{last}}) }{ e_l(t_{\\text{last}}) }

    where :math:`t_{\\text{last}}` is the average of all values of the follower gas at the latest time it
    appears in the database, and the lower case :math:`e` represents the infiller
    database.
    """

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
            ``variable_follower`` (e.g. ``["Emissions|CO2"]``). Note that the 'lead
            gas' methodology gives the same result, independent of the value of
            ``variable_leaders`` in the database.

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
        iamdf_follower = self._get_iamdf_follower(variable_follower, variable_leaders)
        data_follower = iamdf_follower.data

        data_follower_time_col = iamdf_follower.time_col
        data_follower_key_timepoint = max(data_follower[data_follower_time_col])
        key_timepoint_filter = {data_follower_time_col: [data_follower_key_timepoint]}
        data_follower_key_year_val = np.nanmean(
            iamdf_follower.filter(**key_timepoint_filter)["value"].values
        )
        data_follower_unit = data_follower["unit"].values[0]

        if data_follower_time_col == "time":
            data_follower_key_timepoint = data_follower_key_timepoint.to_pydatetime()

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
            lead_var = in_iamdf.filter(variable=variable_leaders)

            if data_follower_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(
                        data_follower_time_col
                    )
                )

            def get_values_in_key_timepoint(idf):
                # filter warning about empty data frame as we handle it ourselves
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return idf.filter(**key_timepoint_filter)

            lead_var_val_in_key_timepoint = get_values_in_key_timepoint(lead_var)

            if lead_var_val_in_key_timepoint.data.empty:
                if not interpolate:
                    error_msg = (
                        "Required downscaling timepoint ({}) is not in the data for "
                        "the lead gas ({})".format(
                            data_follower_key_timepoint, variable_leaders[0]
                        )
                    )
                    raise ValueError(error_msg)
                else:
                    # TODO: make interpolate method of IamDataFrame datetime friendly
                    lead_var_interp = lead_var.timeseries()
                    lead_var_interp[data_follower_key_timepoint] = np.nan
                    lead_var_interp = lead_var_interp.reindex(
                        sorted(lead_var_interp.columns), axis=1
                    )
                    lead_var_interp = IamDataFrame(
                        lead_var_interp.interpolate(method="index", axis=1)
                    )
                    lead_var_val_in_key_timepoint = get_values_in_key_timepoint(
                        lead_var_interp
                    )

            lead_var_val_in_key_timepoint = lead_var_val_in_key_timepoint.timeseries()
            if not lead_var_val_in_key_timepoint.shape[1] == 1:  # pragma: no cover
                raise AssertionError(
                    "How did filtering for a single timepoint result in more than "
                    "one column?"
                )

            lead_var_val_in_key_timepoint = lead_var_val_in_key_timepoint.iloc[:, 0]

            scaling = data_follower_key_year_val / lead_var_val_in_key_timepoint
            output_ts = (lead_var.timeseries().T * scaling).T.reset_index()

            output_ts["variable"] = variable_follower
            output_ts["unit"] = data_follower_unit
            for col in in_iamdf.extra_cols:
                output_ts[col] = ""

            return IamDataFrame(output_ts)

        return filler

    def _get_iamdf_follower(self, variable_follower, variable_leaders):
        if len(variable_leaders) > 1:
            raise ValueError(
                "For `DatabaseCruncherLeadGas`, ``variable_leaders`` should only "
                "contain one variable"
            )

        self._check_follower_and_leader_in_db(variable_follower, variable_leaders)

        iamdf_follower = self._db.filter(variable=variable_follower)
        data_follower = iamdf_follower.data
        return iamdf_follower
