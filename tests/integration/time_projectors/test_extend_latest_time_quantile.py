import datetime as dt
import logging
import re

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from silicone.time_projectors import ExtendLatestTimeQuantile
from silicone.utils import _adjust_time_style_to_match

_msa = ["model_a", "scen_a"]
_mb = "model_b"
_ma = "model_a"


class TestDatabaseCruncherExtendLatestTimeQuantile:
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
        res = tcruncher.derive_relationship("Emissions|HFC|C5F12")
        assert callable(res)

    def test_derive_relationship_error_time_col_mismatch(self, test_db):
        tcruncher = self.tclass(test_db)
        infiller_time_col = test_db.time_col
        error_msg = re.escape(
            "`in_iamdf` time column must be the same as the time column used "
            "to generate this filler function (`{}`)".format(infiller_time_col)
        )
        filler = tcruncher.derive_relationship("Emissions|HFC|C5F12")
        test_2 = test_db.timeseries()
        if infiller_time_col == "year":
            test_2.columns = test_2.columns.map(lambda x: dt.datetime(x, 1, 1))
            test_2 = IamDataFrame(test_2)
        else:
            test_2.columns = test_2.columns.map(lambda x: int(x.year))
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
                [
                    [
                        "ma",
                        "sb",
                        "World",
                        "Emissions|HFC|C2F6",
                        "kt C2F6/yr",
                        "",
                        5,
                        2,
                        3,
                    ]
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
            pd.DataFrame(
                [
                    [
                        "ma",
                        "sa",
                        "World",
                        "Emissions|HFC|C2F6",
                        "kt C2F6/yr",
                        "",
                        1,
                        1,
                        2,
                    ],
                    [
                        "ma",
                        "sb",
                        "World",
                        "Emissions|HFC|C2F6",
                        "kt C2F6/yr",
                        "",
                        2,
                        3,
                        3,
                    ],
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
        infiller_df = test_db.append(
            _adjust_time_style_to_match(IamDataFrame(extra_info), test_db)
        )
        tcruncher = self.tclass(infiller_df)
        cruncher = tcruncher.derive_relationship(variable)
        infill_df = test_db.filter(
            **{test_db.time_col: test_db[test_db.time_col][0]}, keep=False
        )
        infilled_filt = cruncher(infill_df)
        infilled_test = cruncher(test_db)
        assert infilled_filt.equals(infilled_test)
        times = [
            time
            for time in infiller_df[infiller_df.time_col].unique()
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
            expected = [1 + (3 - 1) / 2, 2 + (3 - 2) / 2]
        assert np.allclose(infilled_filt.data["value"], expected)
        # Test that the result can be appended without problems.
        infill_df.append(infilled_filt, inplace=True)
        assert infill_df.filter(
            variable=variable, **{infill_df.time_col: times}
        ).equals(infilled_filt)

    @pytest.mark.parametrize("add_col", [None, "extra_col"])
    def test_relationship_usage(self, range_df, sparse_df, add_col):
        variable = "Emissions|CO2"
        if add_col:
            add_col_val = "blah"
            sparse_df = sparse_df.data
            sparse_df[add_col] = add_col_val
        target_df = IamDataFrame(sparse_df)
        tcruncher = self.tclass(range_df)

        filler = tcruncher.derive_relationship(variable)
        res = filler(target_df)

        # Test it comes back on input timepoints
        np.testing.assert_array_equal(
            res.timeseries().columns.values.squeeze(), [2020, 2050]
        )
        # Test that we can append the output to the input
        append_df = target_df.filter(variable=variable).append(res)
        if add_col:
            assert all(append_df[add_col] == add_col_val)
        append_ts = append_df.timeseries()
        # And has the correct results
        np.testing.assert_array_almost_equal(
            append_ts[2020], [min(n + 1, 101) for n in append_ts[2015]]
        )
        np.testing.assert_array_almost_equal(
            append_ts[2050], [min(n + 2, 102) for n in append_ts[2015]]
        )

    def test_random_value_evolution_behaviour(self, range_df, sparse_df, caplog):
        variable = "Emissions|CO2"
        randdf = range_df.timeseries()
        np.random.seed(10)
        randdf = IamDataFrame(randdf * np.random.random() * 2)
        target_df = IamDataFrame(sparse_df)
        tcruncher = self.tclass(randdf)
        filler = tcruncher.derive_relationship(variable)
        with caplog.at_level(
            logging.INFO, logger="silicone.time_projectors.extend_latest_time_quantile"
        ):
            res = filler(target_df)
        assert len(caplog.record_tuples) == 0
        randts = randdf.timeseries()
        target_dfts = target_df.timeseries()
        rests = res.timeseries()
        start_year = 2015
        quants0 = [
            sum(x <= randts[start_year]) / len(randts[start_year])
            for x in target_dfts[start_year]
        ]
        for year in res.year:
            quant1 = [sum(x <= randts[year]) / len(randts[year]) for x in rests[year]]
            assert quants0 == quant1
        # And if we have one time-point missing we should get a warning error message
        tcruncher = self.tclass(randdf.filter(year=2050, scenario="*9*", keep=False))
        filler = tcruncher.derive_relationship(variable)
        with caplog.at_level(
            logging.INFO, logger="silicone.time_projectors.extend_latest_time_quantile"
        ):
            res = filler(target_df)
        assert len(caplog.record_tuples) == 1
        assert (
            caplog.record_tuples[0][2]
            == "The input database may be inconsistent at later times"
        )
        assert res.timeseries().isna().sum().sum() == 0

    def test_time_val_error(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        variable = "Emissions|HFC|C2F6"
        filler = tcruncher.derive_relationship(variable=variable)
        error_msg = re.escape(
            "The infiller database does not extend in time past the target "
            "database, so no infilling can occur."
        )
        with pytest.raises(ValueError, match=error_msg):
            filler(test_db)
        restricted_test = test_db.filter(
            **{test_db.time_col: min(test_db[test_db.time_col])}
        )
        reconstruct = filler(restricted_test)
        assert (
            reconstruct.append(restricted_test)
            .filter(variable=variable)
            .equals(test_db.filter(variable=variable))
        )

    @pytest.mark.parametrize("smoothing", [True, False])
    def test_weighting_results(self, sparse_df, range_df, smoothing):
        # Ensure that duplicating data and giving it a weight of 1/2 makes little
        # difference to the results

        # First check that giving a weight of 1 does nothing.
        to_infill = range_df.filter(
            **{range_df.time_col: range_df[range_df.time_col][0]}
        )
        sparse_df = sparse_df.data
        sparse_df.loc[0::4, "value"] *= 5
        sparse_df.loc[3::4, "value"] += -1
        sparse_df = IamDataFrame(sparse_df)
        variable = "Emissions|CO2"
        tcruncher = self.tclass(sparse_df)
        normal_results = tcruncher.derive_relationship(
            variable=variable, smoothing=smoothing
        )(to_infill)
        weight_1 = {(_ma, "sc_5"): 1, (_ma, "sc_6"): 1}
        norm_weight_results = tcruncher.derive_relationship(
            variable=variable,
            weighting=weight_1,
            smoothing=smoothing,
        )(to_infill)
        assert normal_results.equals(norm_weight_results)

        # Compare these with a run where we have a duplicated datapoint.
        new_scen = "new_scen"
        duplicated_scen = "sc_6"
        duplicate = sparse_df.filter(model=_ma, scenario=duplicated_scen)
        duplicate.rename({"scenario": {duplicated_scen: new_scen}}, inplace=True)
        larger_db = sparse_df.append(duplicate)
        tcruncher_dup = self.tclass(larger_db)
        weights = {(_ma, duplicated_scen): 0.5, (_ma, new_scen): 0.5}
        weighted_results = tcruncher_dup.derive_relationship(
            variable=variable,
            weighting=weights,
            smoothing=smoothing,
        )(to_infill)
        np.allclose(normal_results.data.value, weighted_results.data.value)
        # Also check that the duplication changes the results!
        duplicate_results = tcruncher_dup.derive_relationship(
            variable=variable, smoothing=smoothing
        )(to_infill)
        assert not np.allclose(
            normal_results.data.value,
            duplicate_results.data.value,
        )

    @pytest.mark.parametrize("smoothing", [True, False])
    def test_weighting_at_limits(self, sparse_df, range_df, smoothing):
        # We should get sensible results when infilling results well outside the limit
        # of the input data
        to_infill = range_df.filter(
            **{range_df.time_col: range_df[range_df.time_col][0]}
        )
        # Adjust the data to have a lowest value well below the normal range
        to_infill = to_infill.data
        to_infill.value.iloc[0] = -100
        to_infill = IamDataFrame(to_infill)

        variable = "Emissions|CO2"
        tcruncher = self.tclass(sparse_df)
        weight = {(_ma, "sc_5"): 1, (_ma, "sc_6"): 2}
        results = tcruncher.derive_relationship(
            variable=variable, smoothing=smoothing, weighting=weight
        )(to_infill)
        assert all(np.isfinite(results.data.value))
        if smoothing:
            # If we do smoothing, we limit the results in a fuzzy way, so that the
            # values must be within one range-distance outside the range of values. The
            # lower point of the range is at 0.
            assert (
                max(results.data.value)
                < max(sparse_df.filter(year=2015).data.value) * 2
            )
            assert min(results.data.value) > -max(
                sparse_df.filter(year=2015).data.value
            )
        else:
            assert max(results.data.value) == max(
                sparse_df.filter(year=2015).data.value
            )
            assert min(results.data.value) == min(
                sparse_df.filter(year=2015).data.value
            )
