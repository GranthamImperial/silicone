import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from silicone.time_projectors import ExtendRMSClosest
from silicone.time_projectors.extend_rms_closest import _select_closest

_msa = ["model_a", "scen_a"]


class TestDatabaseCruncherExtendRMSClosest:
    tclass = ExtendRMSClosest
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

    def test_deriver_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|HFC|C5F12")
        assert callable(res)

    def test_get_iamdf_variable(self, test_db):
        tcruncher = self.tclass(test_db)
        variable = "Emissions|HFC|C2F6"
        res = tcruncher._get_iamdf_variable(variable)
        assert res.variable == [variable]

    def test_get_iamdf_variable_missing(self, test_db):
        tcruncher = self.tclass(test_db)
        variable = "Emissions|CO2"
        with pytest.raises(ValueError):
            tcruncher._get_iamdf_variable(variable)

    @pytest.mark.parametrize("add_col", [None, "extra_col"])
    def test_relationship_usage(self, add_col):
        variable = "Emissions|HFC|C2F6"
        range_df = IamDataFrame(
            pd.DataFrame(
                [
                    [
                        "model_b",
                        "scen_" + str(n),
                        "World",
                        variable,
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
            [["model_a", "sc_1", "World", variable, "kt C2F6/yr", 0, 1.5]],
            columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
        )
        if add_col:
            add_col_val = "blah"
            sparse_df[add_col] = add_col_val
        target_df = IamDataFrame(sparse_df)
        tcruncher = self.tclass(range_df)

        filler = tcruncher.derive_relationship(variable)
        res = filler(target_df)

        # Test it comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(), [2020, 2050]
        )

        # Check it has the correct results
        append_df = target_df.filter(variable=variable).append(res)
        append_ts = append_df.timeseries()
        np.testing.assert_array_almost_equal(append_ts[2020], 2)
        np.testing.assert_array_almost_equal(append_ts[2050], 3)


def test_select_closest():
    bad_target = pd.DataFrame(
        [[1, 2, 3, 4]],
        index=pd.MultiIndex.from_arrays(
            [("chartreuse",), (6,), (5,), (1,)],
            names=("model", "scenario", "homogeneity", "variable"),
        ),
    )

    target = pd.DataFrame(
        [[1, 2, 3]],
        index=pd.MultiIndex.from_arrays(
            [("chartreuse",), (6,), (5,), (1,)],
            names=("model", "scenario", "homogeneity", "variable"),
        ),
    )
    possible_answers = pd.DataFrame(
        [[1, 1, 3.1], [1, 2, 3.5], [1, 2, 3.5], [1, 2, 4]],
        index=pd.MultiIndex.from_arrays(
            [
                ("blue", "red", "green", "yellow"),
                (1.5, 1.6, 2, 0.4),
                (1, 1, 1, 1),
                (1, 1, 1, 1),
            ],
            names=("model", "scenario", "homogeneity", "variable"),
        ),
    )

    error_msg = "Target array does not match the size of the searchable arrays"
    with pytest.raises(ValueError, match=error_msg):
        _select_closest(possible_answers, bad_target)

    closest_meta = _select_closest(possible_answers, target)

    assert closest_meta[0] == "red"
    assert closest_meta[1] == 1.6
