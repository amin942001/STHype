import numpy as np


def is_monotonic(array_to_test: np.ndarray) -> bool:
    """Test if a array_to_test is monotone

    Parameters
    ----------
    array_to_test : np.ndarray
        The list to test

    Returns
    -------
    bool
        Return True if array_to_test is monotone
    """
    return np.all(array_to_test[1:] >= array_to_test[:-1]) or np.all(
        array_to_test[1:] <= array_to_test[:-1]
    )


def to_monotonic(array_to_transform: np.ndarray) -> np.ndarray:
    """Transform in place an array to a monotonic array

    Parameters
    ----------
    array_to_transform : np.ndarray
        the array to transform

    Returns
    -------
    np.ndarray
        a monotonic array created from array_transform
    """
    while not is_monotonic(array_to_transform):
        begin = array_to_transform[0]
        end = array_to_transform[-1]
        floor_begin_length = 0
        floor_end_length = 0
        while array_to_transform[floor_begin_length] == begin:
            floor_begin_length += 1
        while array_to_transform[-1 - floor_end_length] == end:
            floor_end_length += 1
        if floor_begin_length < floor_end_length:
            array_to_transform[:floor_begin_length] = array_to_transform[
                floor_begin_length
            ]
        else:
            array_to_transform[-floor_end_length:] = array_to_transform[
                -1 - floor_end_length
            ]
    return array_to_transform


def breakable_into_two_monotonic(array_to_test: np.ndarray, indexes: list[int]) -> int:
    """If an array can be split into two monotonic list at one of the index of indexes,
    return the index relative to indexes, else return -1

    Parameters
    ----------
    array_to_test : np.ndarray
        The array to split
    indexes : list[int]
        the index to split

    Returns
    -------
    int
        if one index split into two monotonic array,
        return the index relative to indexes, else return -1
    """
    for index, index_to_test in enumerate(indexes):
        if is_monotonic(array_to_test[:index_to_test]) and is_monotonic(
            array_to_test[index_to_test:]
        ):
            return index

    return -1
