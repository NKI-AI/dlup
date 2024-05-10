# Copyright (c) dlup contributors
import tempfile

import numpy as np
import pytest
import pyvips
from PIL import Image, ImageColor

from dlup import SlideImage
from dlup.experimental_backends import ImageBackend, OpenSlideSlide
from dlup.utils.pyvips_utils import vips_to_numpy
from dlup.writers import TiffCompression, TifffileImageWriter, _color_dict_to_color_lut

COLORMAP = {
    1: "green",
    2: "red",
    3: "yellow",
}
COLOR_LUT = _color_dict_to_color_lut(COLORMAP)
RGB_COLORMAP = {index: ImageColor.getrgb(color_name) for index, color_name in COLORMAP.items()}


class TestTiffWriter:
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
    def test_tiff_writer(self, shape, target_mpp):
        random_array = np.random.randint(low=0, high=255, size=shape, dtype=np.uint8)
        pil_image = Image.fromarray(random_array, mode="RGB" if len(shape) == 3 else "L")
        mode = pil_image.mode

        if mode == "L":
            size = pil_image.size
        else:
            size = (*pil_image.size, 3)

        with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
            writer = TifffileImageWriter(
                temp_tiff.name,
                size=size,
                mpp=(target_mpp, target_mpp),
                compression=TiffCompression.NONE,
            )

            writer.from_pil(pil_image)
            vips_image = pyvips.Image.new_from_file(temp_tiff.name)
            vips_image_numpy = vips_to_numpy(vips_image)
            if mode == "L":
                vips_image_numpy = vips_image_numpy[..., 0]

            assert np.allclose(np.asarray(pil_image), vips_image_numpy)

            with SlideImage.from_file_path(temp_tiff.name, backend=ImageBackend.PYVIPS) as slide:
                slide_mpp = slide.mpp
                assert np.allclose(slide_mpp, target_mpp)

            # Let's try the same with directly the backend
            with SlideImage.from_file_path(temp_tiff.name, backend=OpenSlideSlide) as slide:
                slide_mpp = slide.mpp
                assert np.allclose(slide_mpp, target_mpp)

            with SlideImage.from_file_path(temp_tiff.name, backend=ImageBackend.OPENSLIDE) as slide:
                slide_mpp = slide.mpp
                assert np.allclose(slide_mpp, target_mpp)

    @pytest.mark.parametrize(
        ["shape", "target_mpp"],
        [
            [(512, 512), 0.25],
        ],
    )
    def test_color_map(self, shape, target_mpp):
        random_array = np.zeros(shape, dtype=np.uint8)
        # Fill regions with 0, 1, 2, 3
        # Top-left region with 0 (already filled since the array is initialized with zeros)
        # Top-right region with 1
        random_array[0:256, 256:512] = 1
        # Bottom-left region with 2
        random_array[256:512, 0:256] = 2
        # Bottom-right region with 3
        random_array[256:512, 256:512] = 3
        pil_image = Image.fromarray(random_array, mode="RGB" if len(shape) == 3 else "L")
        mode = pil_image.mode

        if mode == "L":
            size = pil_image.size
        else:
            size = (*pil_image.size, 3)

        with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
            writer = TifffileImageWriter(
                temp_tiff.name,
                size=size,
                mpp=(target_mpp, target_mpp),
                compression=TiffCompression.NONE,
                colormap=COLORMAP,
            )

            writer.from_pil(pil_image)

            with SlideImage.from_file_path(temp_tiff.name, backend=ImageBackend.PYVIPS) as slide:
                slide_mpp = slide.mpp
                thumbnail = slide.get_thumbnail(size=shape)
                assert np.allclose(slide_mpp, target_mpp)
                top_right = np.asarray(thumbnail).astype(np.uint8)[0:256, 256:512]
                assert np.all(top_right == RGB_COLORMAP[1])
                bottom_left = np.asarray(thumbnail).astype(np.uint8)[256:512, 0:256]
                assert np.all(bottom_left == RGB_COLORMAP[2])
                bottom_right = np.asarray(thumbnail).astype(np.uint8)[256:512, 256:512]
                assert np.all(bottom_right == RGB_COLORMAP[3])

    def test_image_type(self):
        # Test to raise a value error if color_map is defined for an RGB image.
        random_array = np.random.randint(low=0, high=255, size=(512, 512, 3), dtype=np.uint8)
        pil_image = Image.fromarray(random_array, mode="RGB")
        size = (*pil_image.size, 3)

        with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
            writer = TifffileImageWriter(
                temp_tiff.name, size=size, mpp=(0.25, 0.25), compression=TiffCompression.NONE, colormap=COLORMAP
            )
            with pytest.raises(ValueError, match=r"Colormaps only work with integer-valued images \(e.g. masks\)."):
                writer.from_pil(pil_image)
