"""
Time projectors

The classes in this module are used to infer values for a scenario at later times given
the trends before that time.

"""

from .extend_latest_time_quantile import ExtendLatestTimeQuantile  # noqa: F401
from .extend_rms_closest import ExtendRMSClosest  # noqa: F401
from .linear_extender import LinearExtender  # noqa: F401
