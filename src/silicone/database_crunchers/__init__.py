"""
Database crunchers.

The classes within this module can be used to crunch a database of scenarios. Each
'Cruncher' has methods which return functions which can then be used to infill
emissions detail based on a 'lead' emissions timeseries.
"""

from .lead_gas import DatabaseCruncherLeadGas  # noqa: F401
from .RMS_closest import DatabaseCruncherRMSClosest
