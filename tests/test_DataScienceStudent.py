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
        ("str", (0, 1), TypeError),
        (("str", "str"), (0, 1), TypeError),
        ((0, 1), ("str", "str"), TypeError)
    ],
)
def test_invalid_integral_input(data_science_student, x_limits, x_stats, exception):
    """Test that solve_integral raises exceptions for invalid input."""
    with pytest.raises(exception) as exc_info:
        data_science_student.solve_integral(x_limits, x_stats)
    assert (
        str(exc_info.value) == "x_limits must be a Sequence of two numbers."
        or str(exc_info.value) == "x_stats must be a Sequence of two numbers."
    )