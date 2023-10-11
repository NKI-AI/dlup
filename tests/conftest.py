# Copyright (c) dlup contributors

"""Fixtures, hooks and plugins."""
import pytest

from dlup import SlideImage

from .common import OpenSlideImageMock, SlideConfig


@pytest.fixture
def slide_config():
    """Fixture returning a slide."""
    return SlideConfig()


@pytest.fixture
def openslide_image(slide_config):
    """Fixture returning a mock image."""
    return OpenSlideImageMock.from_slide_config(slide_config)


@pytest.fixture
def dlup_wsi(openslide_image):
    """Generate sample SlideImage object to test."""
    return SlideImage(openslide_image, identifier="mock")
