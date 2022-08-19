# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np
import PIL.Image
import pytest
import pyvips

from dlup.utils.pyvips import vips_to_numpy
from dlup.writers import TiffCompression, TifffileImageWriter


@pytest.mark.parametrize("shape", [[1200, 1200], [800, 550, 3]])
def test_tiff_writer(shape):
    random_array = np.random.randint(low=0, high=255, size=(1200, 1200, 3), dtype=np.uint8)
    pil_image = PIL.Image.fromarray(random_array)

    if pil_image.mode == "L":
        size = pil_image.size
    else:
        size = (*pil_image.size, 3)

    with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
        writer = TifffileImageWriter(temp_tiff.name, size=size, mpp=(0.45, 0.45), compression=TiffCompression.NONE)

        writer.from_pil(pil_image)
        vips_image = pyvips.Image.new_from_file(temp_tiff.name)
        assert np.allclose(np.asarray(pil_image), vips_to_numpy(vips_image))
