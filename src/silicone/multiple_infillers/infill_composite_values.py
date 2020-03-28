import logging
from silicone.utils import _construct_consistent_values

logger = logging.getLogger(__name__)


def infill_composite_values(df, composite_dic=None, factors_dic=None):
    """
    Constructs a series of aggregate variables, calculated as the sums of variables
    that have been reported. If given factors terms too, the terms will be multiplied
    by the factors before summing.

    Parameters
    ----------
    df : :obj:`pyam.IamDataFrame`
        Input data from which to construct consistent values. This is assumed to be
        fully infilled. This will not be checked.

    composite_dic : dict {str: list[str] or dict{str: float}}
        Key: The variable names of the composite. Value: The variable names of the
        constituents, which may include wildcards ('*'). Optionally, these may be
        dictionaries of the names to factors, which we multiply the numbers by before
         summing them. Defaults to a list of PFC, HFC, F-Gases, CO2 and Kyoto gases.
    """
    if not composite_dic:
        composite_dic = {
            "Emissions|PFC": [
                "Emissions*|CF4",
                "Emissions*|C2F6",
                "Emissions*|C3F8",
                "Emissions*|C4F10",
                "Emissions*|C5F12",
                "Emissions*|C6F14",
            ],
            "Emissions|HFC": ["Emissions|HFC*"],
            "Emissions|F-Gases": [
                "Emissions|PFC",
                "Emissions|HFC",
                "Emissions|SF6",
            ],
            "Emissions|Kyoto Gases (AR5-GWP100)": [
                "Emissions|CO2",
                "Emissions|CH4",
                "Emissions|N2O",
                "Emissions|F-Gases",
            ],
        }
    # Do a simple test to ensure we have any relevant data. This does not prevent
    # problems from missing subsets of data.
    to_delete = []
    # Disable core logging to prevent warnings about empty filters
    logging.getLogger("pyam.core").setLevel(logging.CRITICAL)
    for composite in composite_dic.keys():
        if df.filter(variable=composite_dic.get(composite)).data.empty:
            logger.warning("No data found for {}".format(composite_dic.get(composite)))
            to_delete.append(composite)
    logging.getLogger("pyam.core").setLevel(logging.WARNING)
    [composite_dic.pop(x) for x in to_delete]
    # If the composite variables already exist, remove them first
    df = df.filter(variable=composite_dic.keys(), keep=False)
    for composite in composite_dic.keys():
        if type(composite_dic.get(composite)) is dict:
            temp_df = df.filter(
                variable=[composite] + list(composite_dic.get(composite).keys())
            )
            for factor in composite_dic.get(composite).keys():
                temp_df.data["value"].loc[temp_df.data["variable"] == factor] = \
                temp_df.data["value"].loc[
                    temp_df.data["variable"] == factor
                    ] * composite_dic.get(composite).get(factor)
            composite_df = _construct_consistent_values(
                composite, composite_dic.get(composite), temp_df
            )
        else:
            composite_df = _construct_consistent_values(
                composite, composite_dic.get(composite), df
            )
        try:
            to_return.append(composite_df)
        except NameError:
            to_return = composite_df
    return to_return
