import pytest
import numpy as np

from src.utils import lu_decomposition


@pytest.mark.parametrize(
    "A, L_expected, U_expected",
    [
        # 2x2 matrix
        ([[1, 2], [3, 4]], [[1, 0], [3, 1]], [[1, 2], [0, -2]]),
        # 3x3 matrix
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 
            [[1, 0, 0], [4, 1, 0], [7, 2, 1]], 
            [[1, 2, 3], [0, -3, -6], [0, 0, 0]]
        )
    ]
)
def test_lu_decomposition(A, L_expected, U_expected):
    L, U = lu_decomposition(A)
    assert np.allclose(L, L_expected)
    assert np.allclose(U, U_expected)


@pytest.mark.parametrize(
    "A, exception",
    [
        # Empty input
        ([], ValueError),
        # Non-square matrix
        ([[1, 2], [3, 4], [5, 6]], ValueError),
        # Invalid data type (string)
        ([[1, 2], [3, "str"]], TypeError)
    ]
)
def test_lu_decomposition_invalid_inputs(A, exception):
    with pytest.raises(exception):
        lu_decomposition(A)
           