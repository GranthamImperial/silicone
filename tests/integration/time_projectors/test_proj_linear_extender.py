import datetime as dt
import re

import datetime
import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from silicone.time_projectors import LinearExtender

_msa = ["model_a", "scen_a"]
_mb = "model_b"
_ma = "model_a"


class TestDatabaseCruncherExtendLatestTimeQuantile:
    tclass = LinearExtender
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
                _mb,
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
    range_df = IamDataFrame(
        pd.DataFrame(
            [
                [
                    "model_b",
                    "scen_" + str(n),
                    "World",
                    "Emissions|CO2",
                    "kt C2F6/yr",
                    n,
                    1 + n,
                    2 + n,
                    3 + n,
                ]
                for n in range(100)
            ],
            columns=[
                "model",
                "scenario",
                "region",
                "variable",
                "unit",
                2010,
                2015,
                2020,
                2050,
            ],
        )
    )
    sparse_df = pd.DataFrame(
        [
            [
                _ma,
                "sc_" + str(n),
                "World",
                "Emissions|CO2",
                "kt C2F6/yr",
                n,
                10 + 10 * n,
            ]
            for n in range(11)
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|HFC|C5F12", gradient=-1)
        assert callable(res)

    def test_derive_relationship_error_time_col_mismatch(self, test_db):
        tcruncher = self.tclass(test_db)
        infiller_time_col = test_db.time_col
        error_msg = re.escape(
            "The times requested must be in the same format as the time column in the "
            "input database"
        )
        filler = tcruncher.derive_relationship("Emissions|HFC|C5F12", gradient=-1)
        test_2 = test_db.timeseries()
        if infiller_time_col == "year":
            test_2.columns = test_2.columns.map(lambda x: dt.datetime(x, 1, 1))
            test_2 = IamDataFrame(test_2)
        else:
            test_2.columns = test_2.columns.map(lambda x: int(x.year))
            test_2 = IamDataFrame(test_2)
        with pytest.raises(ValueError, match=error_msg):
            filler(test_2)

    @pytest.mark.parametrize("times", [None, [2010, 2015]])
    def test_derive_relationship_works_no_info_leader(self, test_db, times):
        # test that crunching works even if there's no data about the lead gas in the
        # database, provided it's given a gradient or correctly formatted year_value
        variable = "Emissions|HFC|C2F6"
        if times:
            tcruncher = self.tclass()
        else:
            tcruncher = self.tclass(test_db.filter(variable=variable, keep=False))
        error_msg = "Provide either a year_value OR gradient"
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable, times=times)
        tcruncher.derive_relationship(variable, gradient=-1, times=times)
        tcruncher.derive_relationship(variable, year_value=(2090, 0), times=times)
        error_msg = "year_value should be a tuple of the year and the value that year."
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable, year_value=(1, 1, 1), times=times)
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable, year_value=7, times=times)

    def test_derive_relationship_error_no_info(self, test_db, test_downscale_df):
        # test that crunching fails if there's no data for the timescale
        tcruncher = self.tclass()
        variable = "Emissions|HFC|C5F12"
        error_msg = (
            "This function must either be given a list of times or a "
            "database of completed scenarios"
        )
        with pytest.raises(ValueError, match=error_msg):
            filler = tcruncher.derive_relationship(variable, gradient=-1)

    def test_derive_relationship_single_line(self, test_db):
        # We test that the formula produces the correct answer when there is only one
        # value in the target dataframe
        variable = "Emissions|HFC|C2F6"

        tcruncher = self.tclass(test_db)
        # We choose a value of the gradient that makes it return the original answer
        cruncher = tcruncher.derive_relationship(variable, gradient=0.3 / 5)
        infill_df = test_db.filter(**{test_db.time_col: test_db[test_db.time_col][0]})
        infilled_filt = cruncher(infill_df)
        if test_db.time_col == "year":
            assert infilled_filt.equals(test_db.filter(year=2015, variable=variable))
            times = [2013]
        else:
            # A leapyear slightly distorts the calculation
            assert np.allclose(
                infilled_filt["value"],
                test_db.filter(variable=variable)["value"].iloc[1],
                atol=0.0005,
            )
            times = [
                test_db["time"][0] + 3 / 5 * (test_db["time"][1] - test_db["time"][0])
            ]
        # Test that years overwrites the database and we can calculate the values for
        # an intermediate point
        cruncher = tcruncher.derive_relationship(variable, gradient=1, times=times)
        infilled_filt = cruncher(infill_df)
        assert np.allclose(infilled_filt["value"], test_db["value"][0] + 3, atol=0.002)

    @pytest.mark.parametrize("grad", ["gradient", "year_val"])
    def test_relationship_usage(self, range_df, sparse_df, grad):
        variable = "Emissions|CO2"
        tcruncher = self.tclass(range_df)
        if grad == "gradient":
            filler = tcruncher.derive_relationship(variable, gradient=1)
        else:
            filler = tcruncher.derive_relationship(variable, year_value=(2050, 0))
        res = filler(sparse_df)
        # Test it comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(), [2020, 2050]
        )
        # Test that we can append the output to the input
        append_df = sparse_df.filter(variable=variable).append(res)
        append_ts = append_df.timeseries()
        # And has the correct results
        if grad == "gradient":
            np.testing.assert_array_almost_equal(
                append_ts[2020], [n + 5 for n in append_ts[2015]]
            )
            np.testing.assert_array_almost_equal(
                append_ts[2050], [n + 35 for n in append_ts[2015]]
            )
        else:
            np.testing.assert_array_almost_equal(
                append_ts[2020], append_ts[2015] * 30 / 35
            )
            np.testing.assert_array_almost_equal(append_ts[2050], 0 * append_ts[2050])

    def test_time_val_warning(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        variable = "Emissions|HFC|C2F6"
        filler = tcruncher.derive_relationship(variable=variable, gradient=1)
        error_msg = re.escape(
            "No times requested are later than the times already in the database"
        )
        with pytest.raises(ValueError, match=error_msg):
            filler(test_db)
        tcruncher_timed = self.tclass()
        times = test_db[test_db.time_col].unique()
        filler_timed = tcruncher_timed.derive_relationship(
            variable=variable, times=times, gradient=1
        )
        with pytest.raises(ValueError, match=error_msg):
            filler_timed(test_db)
        # Then test that it does work with longer times
        if test_db.time_col == "year":
            times = [2500, 4000, 3000]
        else:
            times = [
                datetime.datetime(year=2100, month=1, day=1),
                datetime.datetime(year=2200, month=1, day=1),
                datetime.datetime(year=2250, month=1, day=1),
            ]
        filler_timed = tcruncher_timed.derive_relationship(
            variable=variable, times=times, gradient=0
        )
        res = filler_timed(test_db)
        res_ts = res.timeseries()
        test_ts = test_db.filter(variable=variable).timeseries()
        expected = test_ts[test_ts.columns[-1]]
        for col in res_ts:
            assert np.allclose(res_ts[col], expected)
