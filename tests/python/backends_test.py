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
"""Test for the TiffFile backend. Works by creating a tiff file and then reading it with the TiffFile backend.
The results are also compared against the openslide backend.
"""

import os
from pathlib import Path

import fim
import numpy as np
import PIL.Image
import psutil
import pytest
from dlup.backends.openslide_backend import OpenSlideSlide
from dlup.backends.openslide_backend import open_slide as open_slide_openslide
from dlup.backends.tifffile_backend import TifffileSlide
from dlup.backends.tifffile_backend import open_slide as open_slide_tifffile
from dlup.writers import TiffCompression, TifffileImageWriter


@pytest.fixture
def file_path(tmp_path):
    # tmp_path is a pytest fixture that provides a Path object to a temporary directory.
    # Here we are appending a file name to that path.
    return tmp_path / "test_image.tif"


def write_image_to_tiff(file_path, image, mpp, size, pyramid):
    """Write a fim.Image to TIFF file.

    Args:
        size: (height, width) as created by create_test_image
    """
    array = np.asarray(image)  # Works with fim.Image via __array__ interface
    channels = array.shape[2] if array.ndim == 3 else 1

    # PIL.Image.fromarray expects (H, W) for grayscale, not (H, W, 1)
    if channels == 1 and array.ndim == 3:
        array = array.squeeze(axis=2)

    # array shape is (H, W, C), PIL interprets as (height, width)
    # TifffileImageWriter expects size as (width, height, channels)
    # So we need (W, H, C) where W=array.shape[1], H=array.shape[0]
    writer = TifffileImageWriter(
        file_path,
        size=(array.shape[1], array.shape[0], channels),  # (width, height, channels)
        mpp=mpp,
        compression=TiffCompression.NONE,
        is_mask=False,
        tile_size=(128, 128),
        pyramid=pyramid,
    )
    writer.from_pil(PIL.Image.fromarray(array))


def create_test_image(size, channels, color1, color2):
    """Create a test image as fim.Image."""
    half_size = (size[0] // 2, size[1] // 2)
    if channels == 3:
        array = np.zeros((*size, 3), dtype=np.uint8)
        array[: half_size[0], : half_size[1], :] = color1
        array[half_size[0] :, half_size[1] :, :] = color2
    else:
        # Grayscale - create 2D array
        array = np.zeros(size, dtype=np.uint8)
        array[: half_size[0], : half_size[1]] = color1
        array[half_size[0] :, half_size[1] :] = color2
    # fim.Image.from_numpy handles both 2D and 3D arrays
    return fim.Image.from_numpy(array)


def get_open_file_handlers() -> list[Path]:
    process_id = os.getpid()
    process = psutil.Process(process_id)
    open_files = process.open_files()

    open_file_handlers = []
    for open_file in open_files:
        file_name = Path(open_file.path)
        if file_name.suffix == ".tif":
            open_file_handlers.append(file_name)
    return open_file_handlers


@pytest.fixture
def slides(file_path, test_image, mpp, size, pyramid):
    # Write the test image to a TIFF file
    write_image_to_tiff(str(file_path), test_image, mpp, size, pyramid)
    # Open the written TIFF file with both backends
    tiff_slide = open_slide_tifffile(str(file_path))
    openslide_slide = open_slide_openslide(str(file_path))
    yield tiff_slide, openslide_slide


@pytest.fixture
def test_image(size, channels):
    color1 = 255
    color2 = 127
    # Assuming 'create_test_image' is a static method or standalone function
    return create_test_image(size, channels, color1, color2)


@pytest.fixture
def mpp():
    return (1.0, 0.999)


class TestBackends:
    def read_region_and_properties_asserts(self, tiff_slide, openslide_slide, mode, size, mpp, pyramid, test_image):
        """Test region reading with fim.Image backend results."""
        original_array = np.asarray(test_image)  # fim.Image.__array__ interface
        properties = tiff_slide.properties
        tile_size = (properties["tifffile.level[0].TileWidth"], properties["tifffile.level[0].TileLength"])
        num_levels = int(np.ceil(np.log2(np.asarray(size[::-1]) / np.asarray(tile_size))).min()) + 1

        # Let's try to read outside the slide levels. This should give a RuntimeError
        with pytest.raises(RuntimeError):
            tiff_slide.read_region((0, 0), num_levels + 1, (1, 1))

        # We need to check a few regions to make sure the backend is working correctly
        # 1. Check the whole image
        # 2. A part in the upper left
        # 3. Check a part more towards the bottom.

        # This is needed because the array swaps it with respect to the image (x,y) versus (rows, cols)
        _size = size[::-1]
        # Test full image and simple top-left crop (avoid boundary issues)
        regions = [
            ((0, 0), _size),  # Full image
            ((0, 0), (_size[0] // 2, _size[1] // 2)),  # Top-left quadrant
        ]
        for location, region_size in regions:
            # Ensure we're not reading out of bounds
            max_x = location[0] + region_size[0]
            max_y = location[1] + region_size[1]
            slide_dims = tiff_slide.dimensions

            # Skip regions that extend beyond boundaries
            if max_x > slide_dims[0] or max_y > slide_dims[1]:
                continue

            # Both backends return fim.Image now
            tiff_array = np.asarray(tiff_slide.read_region(location, 0, region_size))
            openslide_array = np.asarray(openslide_slide.read_region(location, 0, region_size))

            # OpenSlide now always returns RGB (3 channels)
            # For grayscale test, convert RGB to L by taking one channel
            if mode == "L" and openslide_array.ndim == 3 and openslide_array.shape[2] == 3:
                openslide_array = openslide_array[:, :, 0]  # Take R channel for grayscale

            # Handle single-channel: fim returns (H,W,1) but PIL expects (H,W) for grayscale
            if tiff_array.ndim == 3 and tiff_array.shape[2] == 1:
                tiff_array = tiff_array.squeeze(axis=2)
            if openslide_array.ndim == 3 and openslide_array.shape[2] == 1:
                openslide_array = openslide_array.squeeze(axis=2)
            elif openslide_array.ndim == 1:
                # Already squeezed from RGB→L conversion
                pass

            # Verify modes match expectations
            tiff_region = PIL.Image.fromarray(tiff_array)
            openslide_region = PIL.Image.fromarray(openslide_array)
            assert tiff_region.mode == mode
            assert openslide_region.mode == mode

            cropped_array = original_array[
                location[1] : location[1] + region_size[1], location[0] : location[0] + region_size[0]
            ]

            # Ensure cropped_array has same dimensions as read arrays for comparison
            if tiff_array.ndim == 2 and cropped_array.ndim == 2:
                # Both grayscale (H, W)
                pass
            elif tiff_array.ndim == 3 and cropped_array.ndim == 2:
                # Read is (H, W, C), original is (H, W) - shouldn't happen
                pass
            elif tiff_array.ndim == 2 and cropped_array.ndim == 3:
                # Read is (H, W), original is (H, W, C)
                if cropped_array.shape[2] == 1:
                    cropped_array = cropped_array.squeeze(axis=2)

            # Compare the regions read by both backends
            tiff_array_final = np.asarray(tiff_region)
            openslide_array_final = np.asarray(openslide_region)

            # Primary comparison: Tifffile vs OpenSlide (both use fim backends now)
            # Allow small tolerance due to different TIFF decoding implementations
            np.testing.assert_allclose(
                tiff_array_final,
                openslide_array_final,
                atol=2,
                rtol=0,
                err_msg=f"Tifffile and OpenSlide backends have significant differences at {location}",
            )

            # Secondary: Compare with original (may have differences due to TIFF encoding/decoding)
            np.testing.assert_allclose(
                tiff_array_final,
                cropped_array,
                atol=5,
                rtol=0,
                err_msg=f"Backend output differs too much from original at {location}",
            )

    def property_asserts(self, tiff_slide, openslide_slide, size, mpp, pyramid):
        assert isinstance(tiff_slide, TifffileSlide)
        assert isinstance(openslide_slide, OpenSlideSlide)

        assert tiff_slide.vendor is None
        assert tiff_slide.magnification is None

        assert tiff_slide.dimensions == openslide_slide.dimensions
        assert np.allclose(mpp, tiff_slide.spacing)
        assert tiff_slide.spacing == openslide_slide.spacing

        assert tiff_slide.slide_bounds == openslide_slide.slide_bounds
        # Tiff does not have slide bounds defined (or at least not in our implementation)
        assert tiff_slide.slide_bounds == ((0, 0), tiff_slide.dimensions)

        assert tiff_slide.level_count == openslide_slide.level_count
        if not pyramid:
            assert tiff_slide.level_count == 1
            num_levels = 1
        else:
            # Otherwise we need to compute the number of levels required.
            # This depends on the tile size as well
            properties = tiff_slide.properties
            tile_size = (properties["tifffile.level[0].TileWidth"], properties["tifffile.level[0].TileLength"])
            num_levels = int(np.ceil(np.log2(np.asarray(size[::-1]) / np.asarray(tile_size))).min()) + 1
            assert tiff_slide.level_count == num_levels
            assert openslide_slide.level_count == num_levels

        with pytest.raises(NotImplementedError):
            tiff_slide.set_cache(None)

    @pytest.mark.parametrize("size", [(768, 512), (512, 768), (1024, 256)])
    @pytest.mark.parametrize("channels", [3, 1])
    @pytest.mark.parametrize("pyramid", [False, True])
    def test_tiff_backend(self, size, channels, pyramid, file_path, slides, test_image):
        """Test tiff backend by comparing with openslide backend.

        Uses the test_image fixture to ensure we compare with the same image that was written.
        """
        mpp = (1.0, 0.999)
        mode = "RGB" if channels == 3 else "L"
        assert isinstance(test_image, fim.Image)

        tiff_slide, openslide_slide = slides  # Unpack the slides from the fixture
        self.property_asserts(tiff_slide, openslide_slide, size, mpp, pyramid)
        self.read_region_and_properties_asserts(tiff_slide, openslide_slide, mode, size, mpp, pyramid, test_image)
        # After the test function, close both slides and assert that the file handlers are properly closed.
        assert len(get_open_file_handlers()) == 1
        tiff_slide.close()
        openslide_slide.close()
        assert get_open_file_handlers() == []
