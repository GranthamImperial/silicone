from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pyam import IamDataFrame

msa = ["model_a", "scen_a"]
TEST_DB = pd.DataFrame(
    [
        msa + ["World", "Emissions|HFC|C5F12", "kt C5F12/yr", np.nan, 3.14],
        msa + ["World", "Emissions|HFC|C2F6", "kt C2F6/yr", 1.2, 1.5],
    ],
    columns=["model", "scenario", "region", "variable", "unit", 2010, 2015],
)

TEST_YEARS = [2010, 2015]
TEST_DTS = [datetime(2010, 6, 17), datetime(2015, 7, 21)]


@pytest.fixture(
    scope="function",
    params=[
        TEST_YEARS,
        TEST_DTS,
        ["2010-06-17", "2015-07-21"],
        ["2010-06-17 00:00:00", "2015-07-21 12:00:00"],
    ],
)
def test_db(request):
    tdf = TEST_DB.copy()
    tdf = tdf.rename({2010: request.param[0], 2015: request.param[1]}, axis="columns")
    df = IamDataFrame(data=tdf)
    yield df


@pytest.fixture(scope="function")
def test_downscale_df(request):
    return IamDataFrame(request.cls.tdownscale_df)
