# Copyright (c) dlup contributors

"""Fixtures, hooks and plugins."""
import pytest

from dlup import SlideImage
from tests.backends.test_openslide_backend import MockOpenSlideSlide
from tests.test_image import _BASE_CONFIG


@pytest.fixture
def dlup_wsi():
    openslide_slide = MockOpenSlideSlide.from_config(_BASE_CONFIG)
    return SlideImage(openslide_slide, internal_handler="vips")


@pytest.fixture
def openslideslide_image():
    return MockOpenSlideSlide.from_config(_BASE_CONFIG)
