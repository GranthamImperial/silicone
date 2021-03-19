import logging
import re

import datetime as dt
import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from base import _DataBaseCruncherTester
from silicone.database_crunchers import ExtendLatestTimeQuantile

_msa = ["model_a", "scen_a"]


class TestDatabaseCruncherLatestTimeRatio(_DataBaseCruncherTester):
    tclass = ExtendLatestTimeQuantile
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", "", np.nan, 3.14],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", "", 1.2, 1.5],
        ],
        columns=[
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
            "meta1",
            2010,
            2015,
        ],
    )
    tdownscale_df = pd.DataFrame(
        [
            ["model_b", "scen_b", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1, 2, 3],
            [
                "model_b",
                "scen_c",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.1,
                2.2,
                2.8,
            ],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12"
        )
        assert callable(res)

    def test_derive_relationship_error_time_col_mismatch(self, test_db):
        tcruncher = self.tclass(test_db)
        infiller_time_col = test_db.time_col
        error_msg = re.escape(
            "`in_iamdf` time column must be the same as the time column used "
            "to generate this filler function (`{}`)".format(
                infiller_time_col
            )
        )
        filler = tcruncher.derive_relationship("Emissions|HFC|C5F12")
        test_2 = test_db.timeseries()
        if infiller_time_col == "year":
            test_2.columns = test_2.columns.map(
                lambda x: dt.datetime(x, 1, 1)
            )
            test_2 = IamDataFrame(test_2)
        else:
            test_2.columns = test_2.columns.map(
                lambda x: int(x.year)
            )
            test_2 = IamDataFrame(test_2)
        with pytest.raises(ValueError, match=error_msg):
            filler(test_2)

    def test_derive_relationship_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable = "Emissions|HFC|C2F6"
        tcruncher = self.tclass(test_db.filter(variable=variable, keep=False))

        error_msg = re.escape(
            "No data for `variable` ({}) in database".format(variable)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable)

    def test_derive_relationship_error_no_info(self, test_db, test_downscale_df):
        # test that crunching fails if there's no data for the gas to downscale to in
        # the database
        test_downscale_df = test_downscale_df.filter(
            variable="Emissions|HFC|C5F12",
            keep=False,
        )
        tcruncher = self.tclass(test_db)
        variable = "Emissions|HFC|C5F12"
        error_msg = re.escape(
            "No data for `variable` ({}) in target database".format(variable)
        )
        filler = tcruncher.derive_relationship(variable)
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)

    @pytest.mark.parametrize(
        "extra_info",
        (
                pd.DataFrame(
                    [["ma", "sb", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", "", 5,
                      2, 3]],
                    columns=[
                        "model",
                        "scenario",
                        "region",
                        "variable",
                        "unit",
                        "meta1",
                        2015,
                        2020,
                        2030,
                    ],
                ),
                pd.DataFrame(
                    [
                        ["ma", "sa", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", "",
                         1, 1, 2],
                        ["ma", "sb", "World", "Emissions|HFC|C2F6", "kt C2F6/yr", "",
                         2, 3, 3],
                    ],
                    columns=[
                        "model",
                        "scenario",
                        "region",
                        "variable",
                        "unit",
                        "meta1",
                        2015,
                        2020,
                        2030,
                    ],
                ),
        ),
    )
    def test_derive_relationship_single_line(self, test_db, extra_info):
        # We test that the formula produces the correct answer when there is only one
        # value in the target dataframe
        variable = "Emissions|HFC|C2F6"
        infiller_df = self._join_iamdfs_time_wrangle(test_db, IamDataFrame(extra_info))
        tcruncher = self.tclass(infiller_df)
        cruncher = tcruncher.derive_relationship(variable)
        infill_df = test_db.filter(
            **{test_db.time_col: test_db[test_db.time_col][0]},
            keep=False
        )
        infilled_filt = cruncher(infill_df)
        infilled_test = cruncher(test_db)
        assert infilled_filt.equals(infilled_test)
        times = [
            time for time in infiller_df[infiller_df.time_col].unique()
            if time > max(infill_df[infill_df.time_col])
        ]
        assert all(infilled_filt[infilled_filt.time_col].unique() == times)
        if len(extra_info) == 1:
            # If there is only one row in the infiller dataframe, we return that row.
            expected = IamDataFrame(extra_info).filter(year=[2020, 2030]).data["value"]
        else:
            # If there are multiple, we must consider where we lie in between them.
            # With values 1, and 2 in 2015 the input value, 1.5, is halfway through the
            # data. We therefore expect values 2/3rds between the values
            # at each time.
            expected = [1 + (3-1)/2, 2 + (3-2)/2]
        assert np.allclose(infilled_filt.data["value"], expected)
        # Test that the result can be appended without problems.
        infill_df.append(infilled_filt, inplace=True)
        assert infill_df.filter(variable=variable, **{infill_df.time_col:times}).equals(
            infilled_filt
        )

    @pytest.mark.parametrize("add_col", [None, "extra_col"])
    def test_relationship_usage(self, test_db, test_downscale_df, add_col):
        tcruncher = self.tclass(test_db)
        lead = ["Emissions|HFC|C2F6"]
        follow = "Emissions|HFC|C5F12"
        filler = tcruncher.derive_relationship(follow, lead)
        if add_col:
            add_col_val = "blah"
            test_downscale_df = test_downscale_df.data
            test_downscale_df[add_col] = add_col_val
            test_downscale_df = IamDataFrame(test_downscale_df)
            assert test_downscale_df.extra_cols[0] == add_col
        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        res = filler(test_downscale_df)

        lead_iamdf = test_downscale_df.filter(variable=lead)
        lead_val_2015 = lead_iamdf.filter(year=2015).timeseries().values.squeeze()

        exp = (lead_iamdf.timeseries().T * 3.14 / lead_val_2015).T
        exp = exp.reset_index()
        exp["variable"] = follow
        exp["unit"] = "kt C5F12/yr"
        if add_col:
            exp[add_col] = add_col_val
        exp = IamDataFrame(exp)

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            test_downscale_df.timeseries().columns.values.squeeze(),
        )

        # Test that we can append the output to the input
        append_df = test_downscale_df.filter(variable=lead).append(res)
        assert append_df.filter(variable=follow).equals(res)
        if add_col:
            assert all(append_df[add_col] == add_col_val)

    def test_negative_val_warning(self, test_db, test_downscale_df, caplog):
        # quiet pyam
        caplog.set_level(logging.ERROR, logger="pyam")

        tcruncher = self.tclass(test_db)
        lead = ["Emissions|HFC|C2F6"]
        follow = "Emissions|HFC|C5F12"
        filler = tcruncher.derive_relationship(follow, lead)
        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        with caplog.at_level(logging.INFO, logger="silicone.crunchers"):
            filler(test_downscale_df)
        assert len(caplog.record_tuples) == 0
        test_downscale_df = test_downscale_df.data
        test_downscale_df["value"].iloc[0] = -1
        test_downscale_df = IamDataFrame(test_downscale_df)
        with caplog.at_level(logging.INFO, logger="silicone.crunchers"):
            filler(test_downscale_df)
        assert len(caplog.record_tuples) == 1
        warn_str = "Note that the lead variable {} goes negative.".format(lead)
        assert caplog.record_tuples[0][2] == warn_str

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_relationship_usage_interpolation(
        self, test_db, test_downscale_df, interpolate
    ):
        tcruncher = self.tclass(test_db)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df.filter(year=2015, keep=False), test_db
        )

        required_timepoint = test_db.filter(year=2015).data[test_db.time_col].iloc[0]
        if not interpolate:
            if isinstance(required_timepoint, pd.Timestamp):
                required_timepoint = required_timepoint.to_pydatetime()
            error_msg = re.escape(
                "Required downscaling timepoint ({}) is not in the data for the "
                "lead gas (Emissions|HFC|C2F6)".format(required_timepoint)
            )
            with pytest.raises(ValueError, match=error_msg):
                filler(test_downscale_df, interpolate=interpolate)
            return

        res = filler(test_downscale_df, interpolate=interpolate)

        lead_iamdf = test_downscale_df.filter(
            variable="Emissions|HFC|C2F6", region="World", unit="kt C2F6/yr"
        )
        exp = lead_iamdf.timeseries()

        # will have to make this more intelligent for time handling
        lead_df = lead_iamdf.timeseries()
        lead_df[required_timepoint] = np.nan
        lead_df = lead_df.reindex(sorted(lead_df.columns), axis=1)
        lead_df = lead_df.interpolate(method="index", axis=1)
        lead_val_2015 = lead_df[required_timepoint]

        exp = (exp.T * 3.14 / lead_val_2015).T.reset_index()
        exp["variable"] = "Emissions|HFC|C5F12"
        exp["unit"] = "kt C5F12/yr"
        exp = IamDataFrame(exp)

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            test_downscale_df.timeseries().columns.values.squeeze(),
        )
