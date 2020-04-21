"""
Database crunchers.

The classes within this module can be used to crunch a database of scenarios. Each
'Cruncher' has methods which return functions which can then be used to infill
emissions detail (i.e. calculate 'follower' timeseries) based on 'lead' emissions
timeseries.
"""

from .constant_ratio import ConstantRatio  # noqa: F401
from .equal_quantile_walk import EqualQuantileWalk  # noqa: F401
from .interpolate_specified_scenarios_and_models import (  # noqa: F401
    ScenarioAndModelSpecificInterpolate,
)
from .latest_time_ratio import LatestTimeRatio  # noqa: F401
from .linear_interpolation import LinearInterpolation  # noqa: F401
from .quantile_rolling_windows import QuantileRollingWindows  # noqa: F401
from .rms_closest import RMSClosest  # noqa: F401
from .time_dep_quantile_rolling_windows import (  # noqa: F401
    TimeDepQuantileRollingWindows,
)
from .time_dep_ratio import TimeDepRatio  # noqa: F401
