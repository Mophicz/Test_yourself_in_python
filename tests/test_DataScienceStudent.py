import pytest

import numpy as np


@pytest.mark.parametrize(
    "x_limits, x_stats, exception", 
    [
        # Valid input
        ((-5, 5), (0, 5), None),
        # Empty input
        ((), (), TypeError),
        # Non-sequence input
        (0, (0, 1), TypeError),
        # Too few elements
        ((0,), (0, 1), TypeError),
        # Too many elements
        ((0, 1, 2), (0, 1), TypeError),
        # Invalid data type (string)
        (("str", "str"), (0, 1), TypeError),
    ]
)
def test_solve_integral(data_science_student, x_limits, x_stats, exception):
    """Test that solve_integral raises exceptions for invalid input."""
    if exception:
        with pytest.raises(exception) as exc_info:
            data_science_student.solve_integral(x_limits, x_stats)
        assert (
            str(exc_info.value) == "x_limits must be a Sequence of two numbers."
            or str(exc_info.value) == "x_stats must be a Sequence of two numbers."
        )
    else:
        results = data_science_student.solve_integral(
            x_limits, 
            x_stats, 
            f=lambda x: x ** 2
        )
        assert isinstance(results, dict)
        assert results["mean"] == pytest.approx(25 / 3, abs=1e-2)
        assert results["variance"] == pytest.approx(500 / 9, abs=1e-2)
        assert results["standard_deviation"] == pytest.approx(np.sqrt(500 / 9), abs=1e-2)
        assert results["threshold_70_percent"] == pytest.approx(12.25, abs=1e-2)
        assert results["extrema_x"] == [pytest.approx(0, abs=1e-2)]
    

@pytest.mark.parametrize(
    "A, b",
    [
        # Simple case
        ([[1, 2], [3, 4]], [5, 6]),
        # Negative and decimal values
        ([[1.5, -2.0], [0.5, 1.0]], [-1.0, 2.5]),
        # 4x4 matrix
        (
            [[3, 2, 3, 10], [2, -2, 5, 8], [3, 3, 4, 9], [3, 4, -3, -7]], 
            [4, 1, 3, 2]
        ),
    ]
)
def test_solve_SLE(data_science_student, A, b):
    """Test that solve_SLE raises exceptions for invalid input."""
    result = data_science_student.solve_SLE(A, b)
    assert np.dot(A, result) == pytest.approx(b)
        

@pytest.mark.parametrize(
    "A, b, exception",
    [
        # Empty input
        ([], [], ValueError),
        # Non-square matrix
        ([[1, 2], [3, 4], [5, 6]], [1, 2], ValueError),
        # Invalid data type (string)
        ([[1, 2], [3, "str"]], [1, 2], TypeError),
        # Singular matrix
        ([[1, 2, 3], [4, 5, 6], [7, 8, 9]], [1, 2, 3], ValueError,)
    ]
)
def test_solve_SLE_invalid_inputs(data_science_student, A, b, exception):
    with pytest.raises(exception):
            data_science_student.solve_SLE(A, b)
            