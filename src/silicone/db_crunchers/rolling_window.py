from .base import _DBCruncherBase


class DBCruncherRollingWindow(_DBCruncherBase):
    """Rolling window database cruncher"""

    def __init__(self, arg):
        super(DBCruncherRollingWindow, self).__init__()
        self.arg = arg
