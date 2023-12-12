# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np
import PIL.Image
import pytest
import pyvips

from dlup import SlideImage
from dlup.experimental_backends import ImageBackend
from dlup.utils.pyvips_utils import vips_to_numpy
from dlup.writers import Resampling, TiffCompression, TifffileImageWriter


@pytest.mark.parametrize(
    ["shape", "target_mpp"],
    [
        [(1024, 1024, 3), 0.45],
        [(512, 768, 3), 4],
        [(1024, 512, 3), 0.25],
        [(1024, 1024), 10],
        [(512, 768), 0.8],
        [(1024, 512), 0.7],
    ],
)
@pytest.mark.parametrize("interpolator", [None, Resampling.UNKNOWN, Resampling.LANCZOS])
def test_tiff_writer(shape, target_mpp, interpolator):
    random_array = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)
    pil_image = PIL.Image.fromarray(random_array, mode="RGB" if len(shape) == 3 else "L")
    mode = pil_image.mode

    if mode == "L":
        size = pil_image.size
    else:
        size = (*pil_image.size, 3)

    with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
        try:
            writer = TifffileImageWriter(
                temp_tiff.name,
                size=size,
                mpp=(target_mpp, target_mpp),
                compression=None,
                interpolator=interpolator,
            )
        except ValueError as e:
            if interpolator == Resampling.UNKNOWN:
                assert str(e) == f"Invalid interpolator: {interpolator.name}"
                return
            else:
                raise

        writer.from_pil(pil_image)
        vips_image = pyvips.Image.new_from_file(temp_tiff.name)
        vips_image_numpy = vips_to_numpy(vips_image)
        if mode == "L":
            vips_image_numpy = vips_image_numpy[..., 0]

        assert np.allclose(np.asarray(pil_image), vips_image_numpy)

        # OpenSlide does not read tiff mpp's correctly, so we use pyvips to check.
        # TODO: 4.0.0 does.
        with SlideImage.from_file_path(temp_tiff.name, backend=ImageBackend.PYVIPS) as slide:
            slide_mpp = slide.mpp
            assert np.allclose(slide_mpp, target_mpp)
