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

def select_closest(to_search_df, target_series):
    """
    Finds which row in to_search_array is closest in a root-mean-squared manner to the target array.
    In the event that multiple are equally close, returns first row.

    Parameters
    ----------
    to_search_df: Dataframe
        The rows of this dataframe are the candidate closest vectors

    target_series: series
        The vector to which we want to be close

    Returns
    -------
    :obj:index
        The index of the row which is closest to the target series. The first such row if multiple are equally close.
    """
    if len(target_series.shape) != 1:
        raise ValueError("Target array is multidimensional")
    if target_series.shape[0] != to_search_df.shape[1]:
        raise ValueError("Target array does not match the size of the searchable arrays")
        # TODO: make this work
    closeness = []
    for row in range(to_search_df.shape[0]):
        closeness.append(((target_series - to_search_df.iloc[row]) ** 2).sum())
    # Find the minimum closeness and return the index of it
    to_return = closeness.index(min(closeness))
    return to_search_df.index[to_return]
