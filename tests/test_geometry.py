import pytest

from oscillator_cv.geometry import (
    board_rotation_degrees,
    circular_error_degrees,
    make_letterbox_transform,
    potentiometer_angle_degrees,
    wrap_degrees,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0, 0), (180, -180), (181, -179), (-181, 179), (540, -180)],
)
def test_angle_wrapping(value: float, expected: float) -> None:
    assert wrap_degrees(value) == expected


def test_board_and_potentiometer_angle_conventions() -> None:
    rotation = board_rotation_degrees((50, 90), (50, 10))
    assert rotation == pytest.approx(0.0)
    assert potentiometer_angle_degrees((50, 50), (80, 50), rotation) == pytest.approx(0.0)
    assert potentiometer_angle_degrees((50, 50), (50, 20), rotation) == pytest.approx(90.0)
    assert potentiometer_angle_degrees((50, 50), (50, 80), rotation) == pytest.approx(270.0)
    assert circular_error_degrees(179, -179) == pytest.approx(2.0)


def test_letterbox_coordinate_round_trip() -> None:
    transform = make_letterbox_transform(4000, 3000, 576, 768)
    point = (1234.5, 987.25)
    assert transform.inverse_point(transform.forward_point(point)) == pytest.approx(point)
    assert transform.resized_width == 576
    assert transform.resized_height == 432
    assert transform.offset_y == 168
