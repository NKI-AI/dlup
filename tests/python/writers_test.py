# Copyright 2025 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import tempfile

import numpy as np
import pytest
import fim
from dlup import SlideImage
from dlup.backends import OpenSlideSlide
from dlup.utils.backends import ImageBackend
from dlup import Resampling
from dlup.writers import LibtiffImageWriter, TiffCompression, TifffileImageWriter, _color_dict_to_color_lut
from PIL import Image, ImageColor

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
        array = (255 * np.arange(np.prod(shape)) / np.prod(shape)).astype(np.uint8).reshape(shape)
        pil_image = Image.fromarray(array, mode="RGB" if len(shape) == 3 else "L")
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
                pyramid=True,
            )

            writer.from_pil(pil_image)
            fim_image = fim.Image.from_libtiff(temp_tiff.name).to_numpy()
            expected = np.asarray(pil_image)
            if expected.ndim == 2 and fim_image.ndim == 3 and fim_image.shape[-1] == 1:
                expected = expected[..., np.newaxis]
            assert np.allclose(expected, fim_image)

            # Let's try the same with directly the backend
            with SlideImage.from_file_path(temp_tiff.name, backend=OpenSlideSlide) as slide:
                slide_mpp = slide.mpp
                assert np.allclose(slide_mpp, target_mpp)

            with SlideImage.from_file_path(temp_tiff.name, backend=ImageBackend.OPENSLIDE) as slide:
                slide_mpp = slide.mpp
                assert np.allclose(slide_mpp, target_mpp)

    @pytest.mark.parametrize("writer_class", [TifffileImageWriter, LibtiffImageWriter])
    @pytest.mark.parametrize("pyramid", [True, False])
    def test_tiff_writer_pyramid(self, writer_class, pyramid):
        shape = (1010, 2173, 3)
        target_mpp = 1.0
        tile_size = (128, 128)

        array = (255 * np.arange(np.prod(shape)) / np.prod(shape)).astype(np.uint8).reshape(shape)
        pil_image = Image.fromarray(array, mode="RGB")
        size = (*pil_image.size, 3)

        with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
            writer = writer_class(
                temp_tiff.name,
                size=size,
                mpp=(target_mpp, target_mpp),
                tile_size=tile_size,
                compression=TiffCompression.NONE,
                pyramid=pyramid,
            )

            writer.from_pil(pil_image)
            fim_image = fim.Image.from_libtiff(temp_tiff.name)
            n_pages = fim_image.properties.get("num_pages", 0)

            if pyramid:
                if writer_class == TifffileImageWriter:
                    assert n_pages == int(np.ceil(np.log2(np.asarray(size[:-1]) / np.asarray([tile_size]))).min()) + 1
                else:
                    assert n_pages == int(np.ceil(np.log2(np.asarray(size[:-1]) / np.asarray([tile_size]))).max())
            else:
                assert n_pages == 1
            assert fim_image.properties.get("x_res", 0) == 1000.0 and fim_image.properties.get("y_res", 0) == 1000.0

            for page in range(1, n_pages):
                fim_page = fim.Image.from_libtiff(temp_tiff.name, page=page)
                expected_width = size[0] // (2**page)
                expected_height = size[1] // (2**page)
                assert fim_page.dimensions[0] == expected_width
                assert fim_page.dimensions[1] == expected_height

    @pytest.mark.parametrize(
        ["shape", "target_mpp"],
        [
            [(512, 512), 0.25],
        ],
    )
    def test_color_map(self, shape, target_mpp):
        array = np.zeros(shape, dtype=np.uint8)
        # Fill regions with 0, 1, 2, 3
        # Top-left region with 0 (already filled since the array is initialized with zeros)
        # Top-right region with 1
        array[0:256, 256:512] = 1
        # Bottom-left region with 2
        array[256:512, 0:256] = 2
        # Bottom-right region with 3
        array[256:512, 256:512] = 3
        pil_image = Image.fromarray(array, mode="RGB" if len(shape) == 3 else "L")
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

            with SlideImage.from_file_path(
                temp_tiff.name, backend=ImageBackend.OPENSLIDE, interpolator=Resampling.NEAREST
            ) as slide:
                slide_mpp = slide.mpp
                data = slide.read_region((0, 0), 1.0, (512, 512))
                assert np.allclose(slide_mpp, target_mpp)
                top_right = data.to_numpy().astype(np.uint8)[0:256, 256:512]
                assert np.all(top_right == RGB_COLORMAP[1])
                bottom_left = data.to_numpy().astype(np.uint8)[256:512, 0:256]
                assert np.all(bottom_left == RGB_COLORMAP[2])
                bottom_right = data.to_numpy().astype(np.uint8)[256:512, 256:512]
                assert np.all(bottom_right == RGB_COLORMAP[3])

    def test_image_type(self):
        # Test to raise a value error if color_map is defined for an RGB image.
        shape = (512, 512, 3)
        array = (255 * np.arange(np.prod(shape)) / np.prod(shape)).astype(np.uint8).reshape(shape)
        pil_image = Image.fromarray(array, mode="RGB")
        size = (*pil_image.size, 3)

        with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
            writer = TifffileImageWriter(
                temp_tiff.name, size=size, mpp=(0.25, 0.25), compression=TiffCompression.NONE, colormap=COLORMAP
            )
            with pytest.raises(ValueError, match=r"Colormaps only work with integer-valued images \(e.g. masks\)."):
                writer.from_pil(pil_image)

    def test_tifffile_writer_pyramid_small_image_does_not_crash(self):
        # Regression test: when the target image is smaller than the tile size,
        # the pyramid-level computation used to produce a negative count and crash
        # with "IndexError: list index out of range".
        size = (200, 200)  # (width, height)
        tile_size = (512, 512)
        tile = np.zeros((size[1], size[0]), dtype=np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".tiff") as temp_tiff:
            writer = TifffileImageWriter(
                temp_tiff.name,
                size=size,
                mpp=(1.0, 1.0),
                tile_size=tile_size,
                compression=TiffCompression.NONE,
                pyramid=True,
            )
            writer.from_tiles_iterator(iter([tile]))

            vips_image = pyvips.Image.new_from_file(temp_tiff.name)
            assert vips_image.get("n-pages") == 1
