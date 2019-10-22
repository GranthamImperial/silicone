from silicone.utils import add_example, select_closest
import pandas as pd
import pytest

def test_addition():
    expected = 4
    result = add_example(1, 3)

    assert expected == result


def test_select_closest():
    target = pd.Series([1, 2, 3])
    bad_answer = pd.DataFrame([1, 1])
    possible_answers = pd.DataFrame([[1, 1, 1], [1, 2, 3.5], [1, 2, 3.5], [1, 2, 4]])

    with pytest.raises(ValueError, match="Target array is multidimensional"):
        select_closest(bad_answer, bad_answer)

    with pytest.raises(ValueError, match="Target array does not match the size of the searchable arrays"):
        select_closest(bad_answer, target)

    index_of_answer = select_closest(possible_answers, target)
    assert index_of_answer == 1
