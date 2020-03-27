"""
Database crunchers.

The classes within this module can be used to crunch a database of scenarios. Each
'Cruncher' has methods which return functions which can then be used to infill
emissions detail (i.e. calculate 'follower' timeseries) based on 'lead' emissions
timeseries.
"""

from .constant_ratio import DatabaseCruncherConstantRatio  # noqa: F401
from .lead_gas import DatabaseCruncherLeadGas  # noqa: F401
from .quantile_rolling_windows import (  # noqa: F401
    DatabaseCruncherQuantileRollingWindows,
)
from .rms_closest import DatabaseCruncherRMSClosest  # noqa: F401
from .time_dep_ratio import DatabaseCruncherTimeDepRatio  # noqa: F401
from .linear_interpolation import DatabaseCruncherLinearInterpolation  # noqa: F401
from .interpolate_specified_scenarios_and_models import (
    DatabaseCruncherScenarioAndModelSpecificInterpolate,
)  # noqa: F401
