import re
from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

from silicone.database_crunchers import TimeDepQuantileRollingWindows

_ma = "model_a"
_mb = "model_b"
_mc = "model_c"
_sa = "scen_a"
_sb = "scen_b"
_sc = "scen_c"
_sd = "scen_d"
_se = "scen_e"
_eco2 = "Emissions|CO2"
_gtc = "Gt C/yr"
_ech4 = "Emissions|CH4"
_mtch4 = "Mt CH4/yr"
_ec5f12 = "Emissions|HFC|C5F12"
_ktc5f12 = "kt C5F12/yr"
_ec2f6 = "Emissions|HFC|C2F6"
_ktc2f6 = "kt C2F6/yr"
_msrvu = ["model", "scenario", "region", "variable", "unit"]


class TestDatabaseTimeDepCruncherRollingWindows:
    tclass = TimeDepQuantileRollingWindows
    # The units in this dataframe are intentionally illogical for C5F12
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
            [_ma, _sa, "World", _ec5f12, _mtch4, 3.14, 4, 5, 6],
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
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CH4"], {})
        assert callable(res)

    def test_derive_relationship_with_nans(self):
        tdb = self.tdb.copy()
        tdb.loc[(tdb["variable"] == _eco2) & (tdb["model"] == _ma), 2050] = np.nan
        tcruncher = self.tclass(IamDataFrame(tdb))
        res = tcruncher.derive_relationship("Emissions|CO2", ["Emissions|CH4"], {})
        # just make sure that this runs through and no error is raised
        assert callable(res)

    def test_derive_relationship_wrong_times_db(self):
        tdb = self.tdb.copy()
        tcruncher = self.tclass(IamDataFrame(tdb))
        error_msg = re.escape(
            "Not all required times in the dictionary have data in the database."
        )
        times_quant = {2011: 0.5}
        with pytest.raises(ValueError, match=error_msg):
            tcruncher.derive_relationship(
                "Emissions|CO2", ["Emissions|CH4"], times_quant
            )

    def test_derive_relationship_wrong_times_input(self, test_db):
        tcruncher = self.tclass(test_db)
        # Due to a current bug in pyam, we need to ensure this is not a 64-type value
        t_0 = list(test_db[test_db.time_col].unique())[0]
        times_quant = {t_0: 0.5}
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CH4"], times_quant
        )
        error_msg = re.escape(
            "Not all required times in the infillee database can be found in "
            "the dictionary."
        )
        if test_db.time_col == "year":
            filtered_db = test_db.filter(year=int(t_0), keep=False)
        else:
            filtered_db = test_db.filter(
                time=tcruncher._convert_dt64_todt(t_0), keep=False
            )
        with pytest.raises(ValueError, match=error_msg):
            res(filtered_db)

    @pytest.mark.parametrize("timecol", ["year", "time"])
    def test_relationship_usage(self, timecol):
        # Define a regular set of values and check that we return the correct quantiles
        # of it
        if timecol == "year":
            dates = [2010, 2020, 2030]
        else:
            dates = [
                datetime(year=2010, month=6, day=6),
                datetime(year=2020, month=6, day=6),
                datetime(year=2030, month=6, day=6),
            ]
        regular_db = pd.DataFrame(
            [
                [_ma, _sa + str(val), "World", _eco2, _gtc, val, val, val]
                for val in range(11)
            ],
            columns=_msrvu + dates,
        )
        regular_data = IamDataFrame(regular_db)
        regular_data["variable"] = _ech4
        regular_data["value"] = 1
        regular_data.append(IamDataFrame(regular_db), inplace=True)
        regular_data = IamDataFrame(regular_data.data)
        tcruncher = self.tclass(regular_data)
        follow = _eco2
        formatted_dates = list(regular_data[regular_data.time_col].unique())
        quant = {
            formatted_dates[0]: 0.4,
            formatted_dates[1]: 0.5,
            formatted_dates[2]: 0.6,
        }
        res = tcruncher.derive_relationship(
            follow, [_ech4], time_quantile_dict=quant, nwindows=2,
        )
        to_infill = regular_data.filter(variable=_ech4)
        returned = res(to_infill)
        for time, quantile in quant.items():
            if timecol == "year":
                filtered_ans = returned.filter(year=int(time))["value"]
            else:
                filtered_ans = returned.filter(time=tcruncher._convert_dt64_todt(time))[
                    "value"
                ]
            assert np.allclose(filtered_ans, 11 * (quantile - 1 / 22))

    def test_derive_relationship_same_gas(self, test_db):
        # Given only a single data series, we recreate the original pattern for any
        # quantile
        test_db_redux = test_db.filter(scenario=_sa, model=_ma)
        times = list(test_db_redux[test_db_redux.time_col].unique())
        tcruncher = self.tclass(test_db_redux)
        quantile_dict = {times[0]: 0.4, times[1]: 0.9, times[2]: 0.01, times[3]: 0.99}
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CO2"], quantile_dict,
        )
        crunched = res(test_db_redux)
        assert np.allclose(
            crunched["value"], test_db_redux.filter(variable="Emissions|CO2")["value"]
        )

    def test_derive_relationship_int_years(self):
        # The code should be able to use ether int or int64 years
        times = [2010, 2020, 2030]
        regular_db = IamDataFrame(
            pd.DataFrame(
                [
                    [_ma, _sa + str(val), "World", _eco2, _gtc, val, val, val]
                    for val in range(4)
                ],
                columns=_msrvu + times,
            )
        )
        tcruncher = self.tclass(regular_db)
        quantile_dict = {times[0]: 0.4, times[1]: 0.9, times[2]: 0.01}
        res = tcruncher.derive_relationship(
            "Emissions|CO2", ["Emissions|CO2"], quantile_dict,
        )
        crunched = res(regular_db)
        assert len(crunched["value"]) == len(
            regular_db.filter(variable="Emissions|CO2")["value"]
        )
