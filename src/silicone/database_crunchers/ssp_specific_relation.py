"""
Module for the database cruncher which makes a linear interpolator from a subset of scenarios
"""

import numpy as np
import pandas as pd
import scipy.interpolate
from pyam import IamDataFrame

from .base import _DatabaseCruncher
from ..utils import _get_unit_of_variable


class DatabaseCruncherSSPSpecificRelation(_DatabaseCruncher):
    """
    Database cruncher which pre-filters to only use data from specific scenarios, then
    makes a linear interpolator to return values from that set of scenarios. Uses mean
    values in the case of repeated leader values. Returns the follower values at the
    extreme leader values for leader values more extreme than that found in the input
    data.

    """

    def _find_matching_scenarios(
        self,
        to_compare_df,
        variable_follower,
        variable_leaders,
        time_col,
        classify_scenarios,
    ):
        """
        Groups scenarios into different classifications and uses those to work out which
        group contains a trendline most similar to the data.
        In the event of a tie, it returns the scenario name that occurs earlier in the
        input data.

        Parameters
        ----------
        to_compare_df : :obj:`pyam.IamDataFrame`
            The dataframe we wish to find the scenario group closest to. May contain one
            or more scenarios, we minimise the least squared errors for all the data
            colleectively.

        variable_follower : str
            The variable we want to interpolate and compare to the value in to_compare_df

        variable_leaders : list[str]
            The variable(s) we want to use to construct the interpolation
            (e.g. ``["Emissions|CO2"]``). In the event that there are multiple, we
            interpolate with each one separately and minimise the sum of the squared
            errors.

        classify_scenarios : list[str]
            The names of scenarios or groups of scenarios that are possible matches.
            This may have *s to represent wild cards, hence multiple scenarios will have
            all their data combined to make the interpolator.

        Returns
        -------
        String
            The scenario-specifying string that best matches the data.

        Raises
        ------
        ValueError
            Not all required timepoints are present in the database we crunched, we have
             `{dates we have}` but you passed in `{dates we need}`."
        """
        assert (
            len(classify_scenarios) > 1
        ), "There must be multiple options for classify_scenario"
        assert all(
            x in self._db.variables().values
            for x in [variable_follower] + variable_leaders
        ), "Not all required data is present in compared series"
        times_needed = set(to_compare_df.data[time_col])
        if any(x not in self._db.data[time_col].values for x in times_needed):
            raise ValueError(
                "Not all required timepoints are present in the database we "
                "crunched, we have `{}` but you passed in {}.".format(
                    list(set(self._db.data[time_col])),
                    list(set(to_compare_df.data[time_col])),
                )
            )

        scenario_rating = {}
        time_col = self._db.time_col
        convenient_compare_db = self._make_wide_db(to_compare_df).reset_index()
        for scenario in classify_scenarios:
            scenario_db = self._db.filter(
                scenario=scenario, variable=variable_leaders + [variable_follower]
            )
            if scenario_db.data.empty:
                scenario_rating[scenario] = np.inf
                print("Warning: scenario {} not found in data".format(scenario))
                continue

            wide_db = self._make_wide_db(scenario_db)
            squared_dif = 0
            for leader in variable_leaders:
                all_interps = self._make_interpolator(
                    variable_follower, leader, wide_db, time_col
                )
                # TODO: consider weighting by GWP* or similar. Currently no sensible weighting.
                for row in convenient_compare_db.iterrows():
                    squared_dif += (
                        row[1][variable_follower]
                        - all_interps[row[1][time_col]](row[1][leader])
                    ) ** 2
            scenario_rating[scenario] = squared_dif
        ordered_scen = sorted(scenario_rating.items(), key=lambda item: item[1])
        return ordered_scen[0][0]

    def _make_interpolator(
        self, variable_follower, variable_leaders, wide_db, time_col
    ):
        derived_relationships = {}
        for db_time, dbtdf in wide_db.groupby(time_col):
            xs = dbtdf[variable_leaders].values.squeeze()
            ys = dbtdf[variable_follower].values.squeeze()
            if xs.shape != ys.shape:
                raise NotImplementedError(
                    "Having more than one `variable_leaders` is not yet implemented"
                )
            if not xs.shape:
                # 0D-array, so we can return a single value
                xs = np.array([xs])
                ys = np.array([ys])
            # Ensure that any duplicates are replaced by their average value
            xs_pandas = pd.Series(xs)
            for x_dup in xs_pandas[xs_pandas.duplicated()]:
                inds = np.asarray(xs == x_dup).nonzero()[0]
                ys[inds[0]] = ys[inds].mean()
                xs = np.delete(xs, inds[1:])
                ys = np.delete(ys, inds[1:])
            xs, ys = map(np.array, zip(*sorted(zip(xs, ys))))
            if xs.shape == (1,):
                # If there is only one point, we must duplicate the data for interpolate
                xs = np.append(xs, xs)
                ys = np.append(ys, ys)
            derived_relationships[db_time] = scipy.interpolate.interp1d(
                xs,
                ys,
                bounds_error=False,
                fill_value=(ys[0], ys[-1]),
                assume_sorted=True,
            )
        return derived_relationships

    def derive_relationship(
        self, variable_follower, variable_leaders, required_scenario="*"
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
            The string which all accepted scenarios are required to match. This may have
            *s to represent wild cards. It defaults to accept all scenarios.

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
        if len(variable_leaders) != 1:
            raise NotImplementedError(
                "Having more than one `variable_leaders` is not yet implemented"
            )
        use_db = self._db.filter(
            scenario=required_scenario,
            variable=[variable_leaders[0], variable_follower],
        )
        if use_db.data.empty:
            raise ValueError(
                "There is no data of the appropriate type in the database."
                " There may be a typo in the SSP option."
            )
        leader_units = _get_unit_of_variable(use_db, variable_leaders)
        follower_units = _get_unit_of_variable(use_db, variable_follower)
        if len(leader_units) == 0:
            raise ValueError(
                "No data for `variable_leaders` ({}) in database".format(
                    variable_leaders
                )
            )
        if len(follower_units) == 0:
            raise ValueError(
                "No data for `variable_follower` ({}) in database".format(
                    variable_follower
                )
            )
        leader_units = leader_units[0]
        use_db_time_col = use_db.time_col
        use_db = self._make_wide_db(use_db)
        interpolators = self._make_interpolator(
            variable_follower, variable_leaders, use_db, use_db_time_col
        )

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`DatabaseCruncherSSPSpecificRelation`.

            Parameters
            ----------
            in_iamdf : :obj:`pyam.IamDataFrame`
                Input data to fill data in

            Returns
            -------
            :obj:`pyam.IamDataFrame`
                Filled in data (without original source data)

            Raises
            ------
            ValueError
                The key db_times for filling are not in ``in_iamdf``.
            """
            if use_db_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(use_db_time_col)
                )

            var_units = _get_unit_of_variable(in_iamdf, variable_leaders)
            if var_units.size == 0:
                raise ValueError(
                    "There is no data for {} so it cannot be infilled".format(
                        variable_leaders
                    )
                )
            var_units = var_units[0]
            lead_var = in_iamdf.filter(variable=variable_leaders)
            assert (
                lead_var["unit"].nunique() == 1
            ), "There are multiple units for the lead variable."
            if var_units != leader_units:
                raise ValueError(
                    "Units of lead variable is meant to be `{}`, found `{}`".format(
                        leader_units, var_units
                    )
                )
            times_needed = set(in_iamdf.data[in_iamdf.time_col])
            if any(x not in interpolators.keys() for x in times_needed):
                raise ValueError(
                    "Not all required timepoints are present in the database we "
                    "crunched, we crunched \n\t`{}`\nbut you passed in \n\t{}".format(
                        list(interpolators.keys()),
                        in_iamdf.timeseries().columns.tolist(),
                    )
                )
            output_ts = lead_var.timeseries()
            for time in times_needed:
                output_ts[time] = interpolators[time](output_ts[time])
            output_ts.reset_index(inplace=True)
            output_ts["variable"] = variable_follower
            output_ts["unit"] = follower_units[0]
            return IamDataFrame(output_ts)

        return filler

    def _make_wide_db(self, use_db):
        """
        Converts an IamDataFrame into a pandas DataFrame that describes the timeseries
        of variables in index-labelled values.
        :param use_db: PyamDataFrame
        :return: pandas DataFrame
        """
        columns = "variable"
        idx = list(set(use_db.data.columns) - {columns, "value", "unit"})
        use_db = use_db.pivot_table(index=idx, columns=columns, aggfunc="sum")
        # make sure we don't have empty strings floating around (pyam bug?)
        use_db = use_db.applymap(lambda x: np.nan if isinstance(x, str) else x)
        use_db = use_db.dropna(axis=0)
        return use_db
