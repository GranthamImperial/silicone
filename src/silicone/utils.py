"""
Utils contains a number of helpful functions that don't belong elsewhere.
"""


# TODO: put this in pyam
def _get_unit_of_variable(df, variable, multiple_units="raise"):
    """
    Get the unit of a variable in ``self._db``

    Parameters
    ----------
    variable : str
        String to use to filter variables

    multiple_units : str
        If ``"raise"``, check that the variable only has one unit and raise an ``AssertionError`` if it has more than one unit.

    Returns
    -------
    list
        List of units for the variable

    Raises
    ------
    AssertionError
        ``multiple_units=="raise"`` and the filter results in more than one unit
    """
    units = df.filter(variable=variable).data["unit"].unique()
    if len(units) == 0:
        raise ValueError("No units are associated with this variable, {}".format(variable))
    if multiple_units == "raise":
        if len(units) > 1:
            raise AssertionError("`{}` has multiple units".format(variable))
    return units
