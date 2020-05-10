import logging
from sys import platform

import matplotlib as mpl

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

logging.getLogger(__name__).addHandler(logging.NullHandler())

# work around matplotlib explosions, thanks
# https://github.com/uber/ludwig/commit/6b948ea9f0b2e78558fb51d2edd4d9c3558ff505
if platform == "darwin":  # OS X
    mpl.use("TkAgg")
