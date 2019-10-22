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
    if multiple_units == "raise":
        if len(units) > 1:
            raise AssertionError("`{}` has multiple units".format(variable))
        return units

    return units


def select_closest(to_search_array, target_array):
    if len(target_array.shape) != 1:
        ValueError("Target array is multidimensional")
    if target_array.shape[0] != to_search_array.shape[1]:
        ValueError("Target array does not match the size of the searchable arrays")
        # TODO: make this work
    closeness = []
    for row in range(to_search_array.shape[0]):
        closeness.append(((target_array-to_search_array.iloc[row])**2).sum())
    # Find the minimum closeness and return the index of it
    to_return = closeness.index(min(closeness))
    return to_search_array.index[to_return]
