"""
Utils contains a number of helpful functions that don't belong elsewhere.
"""


def add_example(a, b):
    """
    Add two numbers

    Parameters
    ----------
    a : float
        First number to add

    b : float
        Second number to add

    Returns
    -------
    float
        Result of adding `a` and `b`
    """
    return a + b

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
