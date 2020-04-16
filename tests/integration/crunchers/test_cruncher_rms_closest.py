import re

import numpy as np
import pandas as pd
import pyam
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame, concat

from silicone.database_crunchers import RMSClosest
from silicone.database_crunchers.rms_closest import _select_closest

_msa = ["model_a", "scen_a"]


class TestDatabaseCruncherRMSClosest(_DataBaseCruncherTester):
    tclass = RMSClosest
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", np.nan, 3.14],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1.2, 1.5],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
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
    larger_df = pd.DataFrame(
        [
            [
                "model_c",
                "scen_b",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.001,
                2,
                3,
            ],
            [
                "model_c",
                "scen_c",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.1,
                2.2,
                2.8,
            ],
            [
                "model_c",
                "scen_d",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.2,
                2.3,
                2.8,
            ],
            [
                "model_c",
                "scen_b",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1,
                2,
                3,
            ],
            [
                "model_c",
                "scen_c",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1.1,
                2.2,
                2.8,
            ],
            [
                "model_c",
                "scen_d",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1.2,
                2.3,
                2.8,
            ],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050],
    )

    bad_df = pd.DataFrame(
        [
            [
                "model_a",
                "scen_a",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1,
                np.nan,
                np.nan,
            ],
            [
                "model_b",
                "scen_d",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1.2,
                2.3,
                2.8,
            ],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2012, 2015, 2050],
    )

    bad_units_df = pd.DataFrame(
        [
            [
                "model_c",
                "scen_b",
                "World",
                "Emissions|HFC|C2F6",
                "Gt C2F6/yr",
                1.001,
                2,
                3,
            ],
            [
                "model_c",
                "scen_c",
                "World",
                "Emissions|HFC|C2F6",
                "Gt C2F6/yr",
                1.1,
                2.2,
                2.8,
            ],
            [
                "model_c",
                "scen_b",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1,
                2,
                3,
            ],
            [
                "model_c",
                "scen_c",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1.1,
                2.2,
                2.8,
            ],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050],
    )

    multiple_units_df = pd.DataFrame(
        [
            [
                "model_c",
                "scen_b",
                "World",
                "Emissions|HFC|C2F6",
                "Gt C2F6/yr",
                1.001,
                2,
                3,
            ],
            [
                "model_c",
                "scen_c",
                "World",
                "Emissions|HFC|C2F6",
                "kt C2F6/yr",
                1.1,
                2.2,
                2.8,
            ],
            [
                "model_c",
                "scen_b",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1,
                2,
                3,
            ],
            [
                "model_c",
                "scen_c",
                "World",
                "Emissions|HFC|C5F12",
                "kt C5F12/yr",
                1.1,
                2.2,
                2.8,
            ],
        ],
        columns=["model", "scenario", "region", "variable", "unit", 2010, 2015, 2050],
    )

    def test_multiple_units_error(self, multiple_units_df, bad_units_df):
        tcruncher = self.tclass(bad_units_df)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        error_msg = "More than one unit detected for input timeseries"
        with pytest.raises(ValueError, match=error_msg):
            filler(multiple_units_df)

    def test_units_error(self, bad_units_df, test_downscale_df):
        tcruncher = self.tclass(bad_units_df)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        error_msg = "Units of lead variable is meant to be {}, found {}".format(
            bad_units_df.data.unit[0], test_downscale_df.data.unit[0]
        )
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)

    def test_relationship_no_infiller_infillee_time_overlap(
        self, bad_df, test_downscale_df
    ):
        odd_times = bad_df.copy()
        odd_times["scenario"].iloc[0] = "scen_d"
        odd_times["model"].iloc[0] = "model_b"
        tcruncher = self.tclass(odd_times)

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        error_msg = "No time series overlap between the original and unfilled data"
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)

    def test_relationship_no_infiller_time_overlap(self, bad_df, test_downscale_df):
        tcruncher = self.tclass(bad_df)
        error_msg = re.escape(
            "No model/scenario overlap between leader and follower data"
        )
        with pytest.raises(ValueError, match=error_msg):
            filler = tcruncher.derive_relationship(
                "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
            )

    def test_relationship_complex_usage(self, larger_df, test_downscale_df):
        tcruncher = self.tclass(larger_df)
        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        res = filler(test_downscale_df)
        np.testing.assert_allclose(
            res.filter(model="model_b", scenario="scen_b")
            .timeseries()
            .values.squeeze(),
            [1, 2, 3],
        )
        np.testing.assert_allclose(
            res.filter(model="model_b", scenario="scen_c")
            .timeseries()
            .values.squeeze(),
            [1.1, 2.2, 2.8],
        )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )
        assert callable(res)

    def test_derive_relationship_error_multiple_lead_vars(self, test_db):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "For `RMSClosest`, ``variable_leaders`` should only " "contain one variable"
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|HFC|C5F12", ["a", "b"])

    def test_derive_relationship_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|HFC|C2F6"]
        tcruncher = self.tclass(test_db.filter(variable=variable_leaders, keep=False))

        error_msg = re.escape(
            "No data for `variable_leaders` ({}) in database".format(variable_leaders)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|HFC|C5F12", variable_leaders)

    def test_derive_relationship_error_no_overlap(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|HFC|C2F6"]
        db_no_overlap_m = test_db.copy()
        db_no_overlap_m["model"].loc[2] = "different model"
        db_no_overlap_m = IamDataFrame(db_no_overlap_m.data)
        tcruncher = self.tclass(db_no_overlap_m)
        error_msg = re.escape(
            "No model/scenario overlap between leader and follower data"
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|HFC|C5F12", variable_leaders)

    def test_derive_relationship_error_no_info_follower(self, test_db):
        # test that crunching fails if there's no data about the follower gas in the
        # database
        variable_follower = "Emissions|HFC|C5F12"
        tcruncher = self.tclass(test_db.filter(variable=variable_follower, keep=False))

        error_msg = re.escape(
            "No data for `variable_follower` ({}) in database".format(variable_follower)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable_follower, ["Emissions|HFC|C2F6"])

    @pytest.mark.parametrize("add_col", [None, "extra_col"])
    def test_relationship_usage(self, test_db, test_downscale_df, add_col):
        tcruncher = self.tclass(test_db)
        lead = ["Emissions|HFC|C2F6"]
        follow = "Emissions|HFC|C5F12"
        filler = tcruncher.derive_relationship(follow, lead)

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        if add_col:
            add_col_val = "blah"
            test_downscale_df[add_col] = add_col_val
            test_downscale_df = IamDataFrame(test_downscale_df.data)
            assert test_downscale_df.extra_cols[0] == add_col
        res = filler(test_downscale_df)

        scen_b_df = test_db.filter(variable="Emissions|HFC|C5F12")
        scen_c_df = test_db.filter(variable="Emissions|HFC|C5F12")
        scen_b_df["model"] = "model_b"
        scen_b_df["scenario"] = "scen_b"
        scen_c_df["model"] = "model_b"
        scen_c_df["scenario"] = "scen_c"
        if add_col:
            scen_c_df[add_col] = add_col_val
            scen_c_df = IamDataFrame(scen_c_df.data)
            scen_b_df[add_col] = add_col_val
            scen_b_df = IamDataFrame(scen_b_df.data)
        exp = concat([scen_b_df, scen_c_df])

        pd.testing.assert_frame_equal(
            res.timeseries(), exp.timeseries(), check_like=True
        )

        # comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(),
            exp.timeseries().columns.values.squeeze(),
        )

        # Test we can append the answer to the original
        appended_df = test_downscale_df.filter(variable=lead).append(res)
        assert appended_df.filter(variable=follow).equals(res)
        if add_col:
            assert all(appended_df.filter(variable=follow)[add_col] == add_col_val)

    def test_relationship_usage_no_overlap(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db.filter(year=2015))

        filler = tcruncher.derive_relationship(
            "Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"]
        )

        test_downscale_df = self._adjust_time_style_to_match(
            test_downscale_df.filter(year=2015, keep=False), test_db
        )

        error_msg = "No time series overlap between the original and unfilled data"
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)


# other select closest tests to write before making public:
#   - what happens if indexes don't align
def test_select_closest():
    target = pd.Series([1, 2, 3])
    possible_answers = pd.DataFrame(
        [[1, 1, 1], [1, 2, 3.5], [1, 2, 3.5], [1, 2, 4]],
        index=pd.MultiIndex.from_arrays(
            [("blue", "red", "green", "yellow"), (1.5, 1.6, 2, 0.4)],
            names=("colour", "height"),
        ),
    )

    closest_meta = _select_closest(possible_answers, target)
    assert closest_meta["colour"] == "red"
    assert closest_meta["height"] == 1.6


def test_select_closest_multi_dimensional_target_error():
    df = pd.DataFrame([1, 1])
    with pytest.raises(ValueError, match="Target array is multidimensional"):
        _select_closest(df, df)


def test_select_closest_wrong_shape_error():
    to_search = pd.DataFrame([[1, 1, 1], [1, 2, 3.5], [1, 2, 3.5], [1, 2, 4]])
    target = pd.Series([1, 1])

    with pytest.raises(
        ValueError,
        match="Target array does not match the size of the searchable arrays",
    ):
        _select_closest(to_search, target)
