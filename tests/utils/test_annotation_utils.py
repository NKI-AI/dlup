# Copyright (c) dlup contributors
import pytest

from dlup.utils.annotations_utils import hex_to_rgb, rgb_to_hex


@pytest.mark.parametrize("rgb", [(0, 0, 0), (255, 10, 255), (255, 127, 0), (0, 28, 0), (0, 0, 255)])
def test_rgb_to_hex_to_rgb(rgb):
    hex_repr = rgb_to_hex(*rgb)
    rgb2 = hex_to_rgb(hex_repr)
    assert rgb == rgb2


def test_fixed_colors():
    assert hex_to_rgb("black") == (0, 0, 0)


def test_exceptions():
    with pytest.raises(ValueError):
        rgb_to_hex(256, 0, 0)
    with pytest.raises(ValueError):
        rgb_to_hex(0, 256, 0)
    with pytest.raises(ValueError):
        rgb_to_hex(0, 0, 256)
    with pytest.raises(ValueError):
        rgb_to_hex(-1, 0, 0)
    with pytest.raises(ValueError):
        rgb_to_hex(0, -1, 0)
    with pytest.raises(ValueError):
        rgb_to_hex(0, 0, -1)
    with pytest.raises(ValueError):
        hex_to_rgb("1234567")
    with pytest.raises(ValueError):
        hex_to_rgb("#1234567")
    with pytest.raises(ValueError):
        hex_to_rgb("#12345")
    with pytest.raises(ValueError):
        hex_to_rgb("#1234")
    with pytest.raises(ValueError):
        hex_to_rgb("#123")
    with pytest.raises(ValueError):
        hex_to_rgb("#12")
    with pytest.raises(ValueError):
        hex_to_rgb("#1")
    with pytest.raises(ValueError):
        hex_to_rgb("#")
    with pytest.raises(ValueError):
        hex_to_rgb("")
