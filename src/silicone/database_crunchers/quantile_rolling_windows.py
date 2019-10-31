"""
Module for the database cruncher which uses the 'rolling windows' technique.
"""
import numpy as np
import pandas as pd
import scipy.interpolate
from pyam import IamDataFrame

from ..utils import _get_unit_of_variable
from .base import _DatabaseCruncher


class DatabaseCruncherQuantileRollingWindows(_DatabaseCruncher):
    """
    Database cruncher which uses the 'rolling windows' technique.

    This cruncher derives the relationship between two variables by performing
    quantile calculations between the follower timeseries and the lead timeseries.
    These calculations are performed at each timestep in the timeseries, independent
    of the other timesteps.

    For each timestep, the lead timeseries axis is divided into multiple evenly spaced
    windows (to date this is only tested on 1:1 relationships but may work with more
    than one lead timeseries). In each window, every data point in the database is
    included. However, the data points receive a weight given by

    .. math::

        w(x, x_{\\text{window}}) = \\frac{1}{1 + (d_n)^2}

    where :math:`w` is the weight and :math:`d_n` is the normalised distance between
    the centre of the window and the data point's position on the lead timeseries axis.

    :math:`d_n` is calculated as

    .. math::

        d_n = \\frac{x - x_{\\text{window}}}{f \\times (\\frac{b}{2})}

    where :math:`x` is the position of the data point on the lead timeseries axis,
    :math:`x_{\\text{window}}` is the position of the centre of the window on the lead
    timeseries axis, :math:`b` is the distance between window centres and :math:`f` is
    a decay factor which controls how much less points away from :math:`x_{\\
    text{window}}` are weighted. If :math:`f=1` then a point which is halfway between
    window centres receives a weighting of :math:`1/2`. Lowering the value of
    :math:`f` cause points further from the window centre to receive less weight.

    With these weightings, the desired quantile of the data is then calculated. This
    calculation is done by sorting the data, and then choosing the first data point
    from the database which sits above or equal to the given quantile, with each data
    point's contribution to the quantile calculation being weighted by its weight. As
    a result, this cruncher limits itself to using data within the distribution and
    will not make any assumptions about the shape of the distribution.

    By varying the quantile, this cruncher can provide ranges of the relationship
    between different variables. For example, it can provide the 90th percentile (i.e.
    high end) of the relationship between e.g. ``Emissions|CH4`` and ``Emissions|CO2``
    or the 50th percentile (i.e. median) or any other arbitrary percentile/quantile
    choice.
    """

    def derive_relationship(
        self,
        variable_follower,
        variable_leaders,
        quantile=0.5,
        nwindows=10,
        decay_length_factor=1,
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

        quantile : float
            The quantile to return in each window.

        nboxes : int
            The number of windows to use when calculating the relationship between the
            follower and lead gases.

        decay_length_factor : float
            Parameter which controls how strongly points away from the window's centre
            should be weighted compared to points at the centre. Larger values give
            points further away increasingly less weight, smaller values give points
            further away increasingly more weight.

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
            There is no data for ``variable_leaders`` or ``variable_follower`` in the
            database.

        ValueError
            ``quantile`` is not between 0 and 1.

        ValueError
            ``nwindows`` is not equivalent to an integer.

        ValueError
            ``decay_length_factor`` is 0.
        """
        self._check_follower_and_leader_in_db(variable_follower, variable_leaders)

        if not (0 <= quantile <= 1):
            error_msg = "Invalid quantile ({}), it must be in [0, 1]".format(quantile)
            raise ValueError(error_msg)

        if int(nwindows) != nwindows:
            error_msg = "Invalid nwindows ({}), it must be an integer".format(nwindows)
            raise ValueError(error_msg)
        nwindows = int(nwindows)

        if np.equal(decay_length_factor, 0):
            raise ValueError("decay_length_factor must not be zero")

        data_leader_unit = _get_unit_of_variable(self._db, variable_leaders)[0]
        data_follower_unit = _get_unit_of_variable(self._db, variable_follower)[0]
        db_time_col = self._db.time_col

        columns = "variable"
        idx = list(set(self._db.data.columns) - {columns, "value", "unit"})
        wide_db = self._db.filter(
            variable=[variable_follower] + variable_leaders
        ).pivot_table(index=idx, columns=columns, aggfunc="sum")
        # make sure we don't have empty strings floating around (pyam bug?)
        wide_db = wide_db.applymap(lambda x: np.nan if isinstance(x, str) else x)
        wide_db = wide_db.dropna(axis=0)

        derived_relationships = {}
        for db_time, dbtdf in wide_db.groupby(db_time_col):
            xs = dbtdf[variable_leaders].values.squeeze()
            ys = dbtdf[variable_follower].values.squeeze()
            if xs.shape != ys.shape:
                raise NotImplementedError(
                    "Having more than one `variable_leaders` is not yet implemented"
                )

            if not xs.shape:
                # 0D-array, make 1D
                xs = np.array([xs])
                ys = np.array([ys])

            step = (max(xs) - min(xs)) / nwindows
            decay_length = step / 2 * decay_length_factor

            sort_order = np.argsort(ys)
            ys = ys[sort_order]
            xs = xs[sort_order]
            if max(xs) == min(xs):
                # We must prevent singularity behaviour if all the points are at the
                # same x value.
                cumsum_weights = np.array([(1 + x) / len(ys) for x in range(len(ys))])

                def same_x_val_workaround(
                    _, ys=ys, cumsum_weights=cumsum_weights, quantile=quantile
                ):
                    return min(ys[cumsum_weights >= quantile])

                derived_relationships[db_time] = same_x_val_workaround
            else:
                # We want to include the max x point, but not any point above it.
                # The 0.99 factor prevents rounding error inclusion.
                window_centers = np.arange(min(xs), max(xs) + step * 0.99, step)

                db_time_table = pd.DataFrame(
                    index=pd.MultiIndex.from_arrays(
                        ([db_time], [quantile]), names=["db_time", "quantile"]
                    ),
                    columns=window_centers,
                )
                db_time_table.columns.name = "window_centers"

                for window_center in window_centers:
                    weights = 1.0 / (1.0 + ((xs - window_center) / decay_length) ** 2)
                    weights /= sum(weights)
                    # We want to calculate the weights at the midpoint of step corresponding
                    # to the y-value.
                    cumsum_weights = np.cumsum(weights)
                    db_time_table.loc[(db_time, quantile), window_center] = min(
                        ys[cumsum_weights >= quantile]
                    )

                derived_relationships[db_time] = scipy.interpolate.interp1d(
                    db_time_table.columns.values.squeeze(),
                    db_time_table.loc[(db_time, quantile), :].values.squeeze(),
                    bounds_error=False,
                    fill_value=(
                        db_time_table.loc[(db_time, quantile)].iloc[0],
                        db_time_table.loc[(db_time, quantile)].iloc[-1],
                    ),
                )

        def filler(in_iamdf):
            """
            Filler function derived from :obj:`DatabaseCruncherQuantileRollingWindows`.

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
            if db_time_col != in_iamdf.time_col:
                raise ValueError(
                    "`in_iamdf` time column must be the same as the time column used "
                    "to generate this filler function (`{}`)".format(db_time_col)
                )

            var_units = _get_unit_of_variable(in_iamdf, variable_leaders)[0]

            if var_units != data_leader_unit:
                raise ValueError(
                    "Units of lead variable is meant to be `{}`, found `{}`".format(
                        data_leader_unit, var_units
                    )
                )

            # check whether we have all the required timepoints or not
            have_all_timepoints = all(
                [c in derived_relationships for c in in_iamdf.timeseries()]
            )

            if not have_all_timepoints:
                raise ValueError(
                    "Not all required timepoints are present in the database we "
                    "crunched, we crunched \n\t`{}`\nbut you passed in \n\t{}".format(
                        list(derived_relationships.keys()),
                        in_iamdf.timeseries().columns.tolist(),
                    )
                )

            # do infilling here
            infilled_ts = in_iamdf.filter(variable=variable_leaders).timeseries()
            for col in infilled_ts:
                infilled_ts[col] = derived_relationships[col](infilled_ts[col])

            infilled_ts = infilled_ts.reset_index()
            infilled_ts["variable"] = variable_follower
            infilled_ts["unit"] = data_follower_unit

            return IamDataFrame(infilled_ts)

        return filler
