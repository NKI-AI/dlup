# Copyright (c) dlup contributors
import tempfile

import numpy as np
import PIL.Image

from dlup._image import Resampling
from dlup.experimental_backends.openslide_backend import OpenSlideSlide
from dlup.experimental_backends.tifffile_backend import TifffileSlide
from dlup.writers import TiffCompression, TifffileImageWriter


class TestTiffBackend:
    # We'll make a tiff with the writer and read it with tifffile to check if the reader works correctly.
    # Writer is tested independently.
    with tempfile.NamedTemporaryFile(suffix=".tif") as temp_tiff:
        # Let's make an array of size 512, 768 with 3 channels and with a square at the top left and bottom right
        # corners.
        size = (512, 768)
        mpp = (1.0, 0.999)  # TODO: Unfortunately we do not support different mpps, so let's get them close
        array = np.zeros((*size, 3), dtype=np.uint8)
        array[:256, :256] = 255
        array[256:, 256:] = 127

        writer = TifffileImageWriter(
            temp_tiff.name,
            size=(*size[::-1], 3),
            mpp=mpp,
            compression=TiffCompression.NONE,
            interpolator=Resampling.NEAREST,
            tile_size=(128, 128),
            pyramid=False,  # TODO: Why doesn't this work?
        )
        image = PIL.Image.fromarray(array, mode="RGB")
        writer.from_pil(image)

        # Now we can read it back using the tifffile backend
        tiff_slide = TifffileSlide(temp_tiff.name)
        # Let's also do the tiff backend to be able to compare
        openslide_slide = OpenSlideSlide(temp_tiff.name)

        assert tiff_slide.dimensions == image.size
        assert tiff_slide.dimensions == openslide_slide.dimensions
        assert np.allclose(mpp, tiff_slide.spacing)
        assert tiff_slide.spacing == openslide_slide.spacing

        # Reading from the top left
        location = (0, 0)
        level = 0
        size = (256 + 128, 256 + 128)

        tiff_region = tiff_slide.read_region((0, 0), 0, size).convert("RGB")
        openslide_region = openslide_slide.read_region((0, 0), 0, size).convert("RGB")

        assert (np.array(tiff_region) == np.array(openslide_region)).all()

        tiff_array = np.asarray(tiff_region)
        assert tiff_array.shape == (256 + 128, 256 + 128, 3)
        assert ((array[:384, :384] - tiff_array) == 0).all()

        tiff_slide.close()
        openslide_slide.close()
