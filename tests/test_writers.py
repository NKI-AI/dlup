# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np
import PIL.Image
import pytest
import pyvips

from dlup import SlideImage
from dlup.utils.pyvips_utils import vips_to_numpy
from dlup.writers import TiffCompression, TifffileImageWriter


@pytest.mark.parametrize(
    ["shape", "target_mpp"],
    [
        [(1024, 1024), 0.45],
        [(512, 512), 4],
        [(2048, 2048), 0.25],
        [(1024, 1024), 10],
        [(512, 512), 0.8],
        [(2048, 2048), 0.7],
    ],
)
def test_tiff_writer(shape, target_mpp):
    random_array = np.random.randint(low=0, high=255, size=(*shape, 3), dtype=np.uint8)
    pil_image = PIL.Image.fromarray(random_array)

    if pil_image.mode == "L":
        size = pil_image.size
    else:
        size = (*pil_image.size, 3)

    with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
        writer = TifffileImageWriter(
            temp_tiff.name, size=size, mpp=(target_mpp, target_mpp), compression=TiffCompression.NONE
        )

        writer.from_pil(pil_image)
        vips_image = pyvips.Image.new_from_file(temp_tiff.name)
        assert np.allclose(np.asarray(pil_image), vips_to_numpy(vips_image))

        with SlideImage.from_file_path(temp_tiff.name) as slide:
            slide_mpp = slide.mpp
            assert np.allclose(slide_mpp, target_mpp)
