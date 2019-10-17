import datetime as dt
import re
from abc import ABCMeta, abstractmethod

import pytest
from pyam import IamDataFrame


class _DataBaseCruncherTester(metaclass=ABCMeta):
    # crunching class to test
    tclass = None

    # dataframe to use when testing deriving relationship
    tdb = None

    # dataframe to use when testing downscaling (i.e. dataframe containing lead gases)
    tdownscale_df = None

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
    def test_derive_relationship(self, test_db):
        """Test that derive relationship returns the expected type"""
        # should be something like this
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Follower gas", ["Lead gas"])
        assert callable(res)

    @abstractmethod
    def test_relationship_usage(self):
        """Test that derived relationship gives expected results when used"""
        pass

    def test_relationship_usage_wrong_time_col(self, test_db, test_downscale_df):
        test_db = test_db.filter(
            variable = ["Emissions|HFC|C5F12", "Emissions|HFC|C2F6"]
        )
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        if test_db.time_col == "year":
            test_downscale_df = test_downscale_df.timeseries()
            test_downscale_df.columns = test_downscale_df.columns.map(
                lambda x: dt.datetime(x, 1, 1)
            )
            test_downscale_df = IamDataFrame(test_downscale_df)

        error_msg = re.escape(
            "`in_iamdf` time column must be the same as the time column used "
            "to generate this filler function (`{}`)".format(test_db.time_col)
        )
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df, interpolate=True)
