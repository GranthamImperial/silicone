import datetime as dt
from abc import ABCMeta, abstractmethod

from pyam import IamDataFrame


class _DataBaseCruncherTester(metaclass=ABCMeta):
    def _join_iamdfs_time_wrangle(self, base, other):
        return base.append(self._adjust_time_style_to_match(other, base))

    def _adjust_time_style_to_match(self, in_df, target_df):
        if in_df.time_col != target_df.time_col:
            in_df = in_df.timeseries()
            if target_df.time_col == "time":
                target_df_year_map = {v.year: v for v in target_df.timeseries().columns}
                in_df.columns = in_df.columns.map(
                    lambda x: target_df_year_map[x]
                    if x in target_df_year_map
                    else dt.datetime(x, 1, 1)
                )
            else:
                in_df.columns = in_df.columns.map(lambda x: x.year)
            return IamDataFrame(in_df)

        return in_df

    @abstractmethod
    def test_relationship_usage(self):
        """Test that derived relationship gives expected results when used"""
        pass
