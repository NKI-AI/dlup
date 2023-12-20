# coding=utf-8
# Copyright (c) dlup contributors
import tempfile

import numpy as np
import pytest
import pyvips
from PIL import Image, ImageColor

from dlup import SlideImage
from dlup.experimental_backends import ImageBackend
from dlup.utils.pyvips_utils import vips_to_numpy
from dlup.writers import TiffCompression, TifffileImageWriter


def _color_dict_to_clut(color_map: dict[int, str]) -> tuple[np.ndarray, dict[int, tuple[int, int, int]]]:
    # Convert color names to RGB values
    rgb_color_map = {index: ImageColor.getrgb(color_name) for index, color_name in color_map.items()}

    # Initialize a 3x256 CLUT (for 8-bit images)
    clut = np.zeros((3, 256), dtype=np.uint16)

    # Prepare indices and corresponding colors for assignment
    indices = np.array(list(rgb_color_map.keys()))
    colors = np.array(list(rgb_color_map.values())) * 256  # Scale to 16-bit color depth

    # Assign colors to clut using advanced indexing
    clut[0, indices] = colors[:, 0]  # Red channel
    clut[1, indices] = colors[:, 1]  # Green channel
    clut[2, indices] = colors[:, 2]  # Blue channel

    return clut, rgb_color_map


color_map = {
    1: "green",  # Green for stroma
    2: "red",  # Red for tumor
    3: "yellow",  # Yellow for ignore
}
clut, rgb_color_map = _color_dict_to_clut(color_map)


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

            # OpenSlide does not read tiff mpp's correctly, so we use pyvips to check.
            # TODO: 4.0.0 does.
            with SlideImage.from_file_path(temp_tiff.name, backend=ImageBackend.PYVIPS) as slide:
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
                colormap=clut,
            )

            writer.from_pil(pil_image)

            with SlideImage.from_file_path(temp_tiff.name, backend=ImageBackend.PYVIPS) as slide:
                slide_mpp = slide.mpp
                thumbnail = slide.get_thumbnail(size=shape)
                assert np.allclose(slide_mpp, target_mpp)
                top_right = np.asarray(thumbnail).astype(np.uint8)[0:256, 256:512]
                assert np.all(top_right == rgb_color_map[1])
                bottom_left = np.asarray(thumbnail).astype(np.uint8)[256:512, 0:256]
                assert np.all(bottom_left == rgb_color_map[2])
                bottom_right = np.asarray(thumbnail).astype(np.uint8)[256:512, 256:512]
                assert np.all(bottom_right == rgb_color_map[3])
