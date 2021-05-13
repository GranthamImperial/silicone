import datetime as dt 
import re 

import numpy as np 
import pandas as pd 
import pytest
from pyam import IamDataFrame 

from silicone.time_projectors import ExtendRMSClosest
from silicone.utils import _adjust_time_style_to_match 

_msa = ["model_a", "scen_a"]

class TestDatabaseCruncherExtendRMSClosest:
    tclass = ExtendRMSClosest
    tdb = pd.DataFrame(
        [
            _msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", "", np.nan, 3.14],
            _msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", "", 1.2, 1.5],
        ],
        columns = [
            "model",
            "scenario",
            "region",
            "variable",
            "unit",
            "meta1",
            2010,
            2015
        ]
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
        assert(callable(res))

    def test_get_iamdf_variable(self, test_db):
        tcruncher = self.tclass(test_db)
        variable = 'Emissions|HFC|C2F6'
        res = tcruncher._get_iamdf_variable(variable)
        assert res.variable == [variable] 

    def test_get_iamdf_variable_missing(self,test_db):
        tcruncher = self.tclass(test_db)
        variable = 'Emissions|CO2'
        with pytest.raises(ValueError):
            tcruncher._get_iamdf_variable(variable)