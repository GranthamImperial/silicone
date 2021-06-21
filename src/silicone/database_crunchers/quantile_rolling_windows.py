"""
Module for the database cruncher which uses the 'rolling windows' technique.
"""

import logging

import numpy as np
import scipy.interpolate
from pyam import IamDataFrame

from ..stats import rolling_window_find_quantiles
from ..utils import _get_unit_of_variable
from .base import _DatabaseCruncher

logger = logging.getLogger(__name__)


class QuantileRollingWindows(_DatabaseCruncher):
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
    a decay factor which controls how much less points away from
    :math:`x_{\\text{window}}` are weighted.
    If :math:`f=1` then a point which is half the width between window centres away
    receives a weighting of :math:`1/2`. Lowering the value of :math:`f` cause points
    further from the window centre to receive less weight.

    With these weightings, the desired quantile of the data is then calculated. This
    calculation is done by sorting the data by the database's follow timeseries values
    (then by lead timeseries values in the case of identical follow values). From here,
    the weight of each point is calculated following the formula given above.
    We calculate the cumulative sum of weights, and then the cumulative sum up to half
    weights, defined by

    .. math::

        c_{hw} = c_w - 0.5 \\times w

    where :math:`c_w` is the cumulative weights and :math:`w` is the raw weights. This
    ensures that quantiles less than half the weight of the smallest follow value return
    the smallest follow value and more than one minus half the weight of the largest
    follow value return the largest value. Without such a shift, the largest value is
    only returned if the quantile is 1, leading to a bias towards smaller values.

    With these calculations, we have determined the relationship between the follow
    timeseries values and the quantile i.e. cumulative sum of (normalised) weights. We
    can then determine arbitrary quantiles by linearly interpolating.

    If the option ``use_ratio`` is set to ``True``, instead of returning the absolute
    value of the follow at this quantile, we return the quantile of the ratio between
    the lead and follow data in the database, multiplied by the actual lead value of the
    database being infilled.

    By varying the quantile, this cruncher can provide ranges of the relationship
    between different variables. For example, it can provide the 90th percentile (i.e.
    high end) of the relationship between e.g. ``Emissions|CH4`` and ``Emissions|CO2``
    or the 50th percentile (i.e. median) or any other arbitrary percentile/quantile
    choice. Note that the impact of this will strongly depend on nwindows and
    decay_length_factor. Using the :class:`TimeDepQuantileRollingWindows` class makes
    it is possible to specify a dictionary of dates to quantiles, in which case we
    return that quantile for that year or date.
    """

    def derive_relationship(
        self,
        variable_follower,
        variable_leaders,
        quantile=0.5,
        nwindows=11,
        decay_length_factor=1,
        use_ratio=False,
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

        nwindows : int
            The number of window centers to use when calculating the relationship
            between the follower and lead gases.

        decay_length_factor : float
            Parameter which controls how strongly points away from the window's centre
            should be weighted compared to points at the centre. Larger values give
            points further away increasingly less weight, smaller values give points
            further away increasingly more weight.

        use_ratio : bool
            If false, we use the quantile value of the weighted mean absolute value. If
            true, we find the quantile weighted mean ratio between lead and follow,
            then multiply the ratio by the input value.

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
            ``nwindows`` is not equivalent to an integer or is not greater than 1.

        ValueError
            ``decay_length_factor`` is 0.
        """
        self._check_follower_and_leader_in_db(variable_follower, variable_leaders)

        if not (0 <= quantile <= 1):
            error_msg = "Invalid quantile ({}), it must be in [0, 1]".format(quantile)
            raise ValueError(error_msg)

        if int(nwindows) != nwindows or nwindows < 2:
            error_msg = "Invalid nwindows ({}), it must be an integer > 1".format(
                nwindows
            )
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

            if use_ratio:
                # We want the ratio between x and y, not the actual values of y.
                ys = ys / xs
                if np.isnan(ys).any():
                    logger.warning(
                        "Undefined values of ratio appear in the quantiles when "
                        "infilling {}, setting some values to 0 (this may not affect "
                        "results).".format(variable_follower)
                    )
                    ys[np.isnan(ys)] = 0

            if np.equal(max(xs), min(xs)):
                # We must prevent singularity behaviour if all the points are at the
                # same x value.
                cumsum_weights = np.array([(0.5 + x) / len(ys) for x in range(len(ys))])
                ys.sort()

                def same_x_val_workaround(
                    _, ys=ys, cumsum_weights=cumsum_weights, quantile=quantile
                ):
                    if np.equal(min(ys), max(ys)):
                        return ys[0]
                    return scipy.interpolate.interp1d(
                        cumsum_weights,
                        ys,
                        bounds_error=False,
                        fill_value=(ys[0], ys[-1]),
                        assume_sorted=True,
                    )(quantile)

                derived_relationships[db_time] = same_x_val_workaround

            else:
                db_time_table = rolling_window_find_quantiles(
                    xs, ys, quantile, nwindows, decay_length_factor
                )

                derived_relationships[db_time] = scipy.interpolate.interp1d(
                    db_time_table.index.values,
                    db_time_table.loc[:, quantile].values.squeeze(),
                    bounds_error=False,
                    fill_value=(
                        db_time_table[quantile].iloc[0],
                        db_time_table[quantile].iloc[-1],
                    ),
                )

        def filler(in_iamdf):
            """
            Filler function derived from :class:`QuantileRollingWindows`.

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

            var_units = _get_unit_of_variable(in_iamdf, variable_leaders)
            if var_units.size == 0:
                raise ValueError(
                    "There is no data for {} so it cannot be infilled".format(
                        variable_leaders
                    )
                )
            var_units = var_units[0]

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

            if use_ratio and (infilled_ts.values < 0).any():
                warn_str = "Note that the lead variable {} goes negative.".format(
                    variable_leaders
                )
                logger.warning(warn_str)
                print(warn_str)

            for col in infilled_ts:
                if use_ratio:
                    infilled_ts[col] = (
                        derived_relationships[col](infilled_ts[col]) * infilled_ts[col]
                    )
                else:
                    infilled_ts[col] = derived_relationships[col](infilled_ts[col])

            infilled_ts = infilled_ts.reset_index()
            infilled_ts["variable"] = variable_follower
            infilled_ts["unit"] = data_follower_unit

            return IamDataFrame(infilled_ts)

        return filler
