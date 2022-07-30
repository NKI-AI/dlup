# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np
import PIL.Image
import pyvips

from dlup.experimental_writers import TiffCompression, TifffileImageWriter
from dlup.utils.pyvips import vips_to_numpy


def test_tiff_writer():
    random_array = np.random.randint(low=0, high=255, size=(1200, 1200, 3), dtype=np.uint8)
    pil_image = PIL.Image.fromarray(random_array)

    writer = TifffileImageWriter(size=np.asarray(pil_image).shape, mpp=(0.45, 0.45), compression=TiffCompression.NONE)
    with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
        writer.from_pil(pil_image, temp_tiff.name)
        vips_image = pyvips.Image.new_from_file(temp_tiff.name)
        assert np.allclose(np.asarray(pil_image), vips_to_numpy(vips_image))


if __name__ == "__main__":
    test_tiff_writer()
