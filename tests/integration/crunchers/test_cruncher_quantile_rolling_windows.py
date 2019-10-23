import datetime as dt
import re

import pandas as pd
import pytest
from base import _DataBaseCruncherTester
from pyam import IamDataFrame

from silicone.database_crunchers import DatabaseCruncherQuantileRollingWindows

_ma = "model_a"
_mb = "model_b"
_mc = "model_c"
_sa = "scen_a"
_sb = "scen_b"
_sc = "scen_c"
_eco2 = "Emissions|CO2"
_gtc = "Gt C/yr"
_ech4 = "Emissions|CH4"
_mtch4 = "Mt CH4/yr"
_ec5f12 = "Emissions|HFC|C5F12"
_ktc5f12 = "kt C5F12/yr"
_ec2f6 = "Emissions|HFC|C2F6"
_ktc2f6 = "kt C2F6/yr"
_msrvu = ["model", "scenario", "region", "variable", "unit"]


class TestDatabaseCruncherRollingWindows(_DataBaseCruncherTester):
    tclass = DatabaseCruncherQuantileRollingWindows
    tdb = pd.DataFrame(
        [
            [_ma, _sa, "World", _eco2, _gtc, 1, 2, 3, 4],
            [_ma, _sb, "World", _eco2, _gtc, 1, 2, 2, 1],
            [_mb, _sa, "World", _eco2, _gtc, 0.5, 3.5, 3.5, 0.5],
            [_mb, _sb, "World", _eco2, _gtc, 3.5, 0.5, 0.5, 3.5],
            [_ma, _sa, "World", _ech4, _mtch4, 100, 200, 300, 400],
            [_ma, _sb, "World", _ech4, _mtch4, 100, 200, 250, 300],
            [_mb, _sa, "World", _ech4, _mtch4, 220, 260, 250, 230],
            [_mb, _sb, "World", _ech4, _mtch4, 50, 200, 500, 800],
            [_ma, _sa, "World", _ec5f12, _ktc5f12, 3.14, 4, 5, 6],
            [_ma, _sa, "World", _ec2f6, _ktc2f6, 1.2, 1.5, 1, 0.5],
        ],
        columns=_msrvu + [2010, 2030, 2050, 2070],
    )
    tdownscale_df = pd.DataFrame(
        [
            [_mc, _sa, "World", _eco2, _gtc, 1, 2, 3, 4],
            [_mc, _sb, "World", _eco2, _gtc, 0.5, 0.5, 0.5, 0.5],
            [_mc, _sc, "World", _eco2, _gtc, 5, 5, 5, 5],
            [_ma, _sc, "World", _eco2, _gtc, 1.5, 2.5, 2.8, 1.8],
        ],
        columns=_msrvu + [2010, 2030, 2050, 2070],
    )

    def test_derive_relationship(self, test_db):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CH4"])
        assert callable(res)

    def test_derive_relationship_same_gas(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        with pytest.warns(UserWarning) as record:
            tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"])

        assert len(record) == 1
        assert record[0].message.args[
            0
        ] == "`derive_relationship` is not fully tested for {}, use with caution".format(
            self.tclass
        )

        # the test should look like this in future, maybe?
        """
        obs = res(test_downscale_df)
        # going to need a plotter before we can work out what is meant to actually
        # happen here...
        pd.testing.assert_frame_equal(
            obs.timeseries(),
            test_downscale_df.filter(variable="Emissions|CO2").timeseries(),
            check_like=True,
        )
        """

    def test_derive_relationship_error_no_info_leader(self, test_db):
        # test that crunching fails if there's no data about the lead gas in the
        # database
        variable_leaders = ["Emissions|CO2"]
        tcruncher = self.tclass(test_db.filter(variable=variable_leaders, keep=False))

        error_msg = re.escape(
            "No data for `variable_leaders` ({}) in database".format(variable_leaders)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship("Emissions|CH4", variable_leaders)

    def test_derive_relationship_error_no_info_follower(self, test_db):
        # test that crunching fails if there's no data about the follower gas in the
        # database
        variable_follower = "Emissions|CH4"
        tcruncher = self.tclass(test_db.filter(variable=variable_follower, keep=False))

        error_msg = re.escape(
            "No data for `variable_follower` ({}) in database".format(variable_follower)
        )
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(variable_follower, ["Emissions|CO2"])

    @pytest.mark.parametrize("quantile", (-0.1, 1.1, 10))
    def test_derive_relationship_error_quantile_out_of_bounds(self, test_db, quantile):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "Invalid quantile ({}), it must be in [0, 1]".format(quantile)
        )

        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|CH4", ["Emissions|CO2"], quantile=quantile
            )

    @pytest.mark.parametrize("nwindows", (1.1, 3.1, 101.2))
    def test_derive_relationship_nwindows_not_integer(self, test_db, nwindows):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape(
            "Invalid nwindows ({}), it must be an integer".format(nwindows)
        )

        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|CH4", ["Emissions|CO2"], nwindows=nwindows
            )

    @pytest.mark.parametrize("decay_length_factor", (0,))
    def test_derive_relationship_error_decay_length_factor_zero(
        self, test_db, decay_length_factor
    ):
        tcruncher = self.tclass(test_db)
        error_msg = re.escape("decay_length_factor must not be zero")

        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|CH4",
                ["Emissions|CO2"],
                decay_length_factor=decay_length_factor,
            )

    def test_relationship_usage_wrong_unit(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"])

        exp_units = test_db.filter(variable="Emissions|CO2")["unit"].iloc[0]

        wrong_unit = "t C/yr"
        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        test_downscale_df["unit"] = wrong_unit

        error_msg = re.escape(
            "Units of lead variable is meant to be `{}`, found `{}`".format(
                exp_units, wrong_unit
            )
        )
        with pytest.raises(ValueError, match=error_msg):
            res(test_downscale_df)

    def test_relationship_usage_wrong_time(self):
        tdb = IamDataFrame(self.tdb)
        tcruncher = self.tclass(tdb)
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CO2"])

        test_downscale_df = IamDataFrame(self.tdb).timeseries()
        test_downscale_df.columns = test_downscale_df.columns.map(
            lambda x: dt.datetime(x, 1, 1)
        )
        test_downscale_df = IamDataFrame(test_downscale_df)

        error_msg = re.escape(
            "`in_iamdf` time column must be the same as the time column used "
            "to generate this filler function (`year`)"
        )
        with pytest.raises(ValueError, match=error_msg):
            res(test_downscale_df)

    def test_relationship_usage_insufficient_timepoints(
        self, test_db, test_downscale_df
    ):
        tcruncher = self.tclass(test_db.filter(year=2030, keep=False))

        filler = tcruncher.derive_relationship("Emissions|CH4", ["Emissions|CO2"])

        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)

        error_msg = re.escape(
            "Not all required timepoints are present in the IamDataFrame to "
            "downscale, we require `{}`".format(test_db.timeseries().columns.tolist())
        )
        with pytest.raises(ValueError, match=error_msg):
            filler(test_downscale_df)

    def test_relationship_usage(self, test_db, test_downscale_df):
        tcruncher = self.tclass(test_db)

        with pytest.warns(UserWarning) as record:
            tcruncher.derive_relationship("Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"])

        assert len(record) == 1
        assert record[0].message.args[
            0
        ] == "`derive_relationship` is not fully tested for {}, use with caution".format(
            self.tclass
        )

        # the test should look like this in future, maybe?
        """
        test_downscale_df = self._adjust_time_style_to_match(test_downscale_df, test_db)
        res = filler(test_downscale_df)

        lead_iamdf = test_downscale_df.filter(variable="Emissions|HFC|C2F6")
        lead_val_2015 = lead_iamdf.filter(year=2015).timeseries().values.squeeze()

        exp = (lead_iamdf.timeseries().T * 3.14 / lead_val_2015).T
        exp = exp.reset_index()
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
        """

    @pytest.mark.parametrize("interpolate", [True, False])
    def test_relationship_usage_interpolation(
        self, test_db, test_downscale_df, interpolate
    ):
        tcruncher = self.tclass(test_db)

        with pytest.warns(UserWarning) as record:
            tcruncher.derive_relationship("Emissions|HFC|C5F12", ["Emissions|HFC|C2F6"])

        assert len(record) == 1
        assert record[0].message.args[
            0
        ] == "`derive_relationship` is not fully tested for {}, use with caution".format(
            self.tclass
        )

        # the test should look like this in future, maybe?
        """
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
        """
