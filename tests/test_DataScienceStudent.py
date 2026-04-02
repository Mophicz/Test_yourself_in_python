import pytest


@pytest.mark.parametrize(
    "x_limits, x_stats, exception", 
    [
        (0, (0, 1), TypeError),
        ((0, 1), 0, TypeError),
        ((0,), (0, 1), TypeError),
        ((0, 1), (0,), TypeError),
        ((0, 1, 2), (0, 1), TypeError),
        ((0, 1), (0, 1, 2), TypeError), 
        ("str", (0, 1), TypeError)
    ],
)


def test_invalid_integral_input(data_science_student, x_limits, x_stats, exception):
    """Test that solve_integral raises exceptions for invalid input."""
    with pytest.raises(exception):
        assert data_science_student.solve_integral(x_limits, x_stats)
        