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
import fim
import numpy as np
from dlup.backends.common import AbstractSlideBackend


class TestAbstractBackend:
    class DummySlideBackend(AbstractSlideBackend):
        # Minimal implementation to avoid the ABC restriction.
        def __init__(self, filename: str):
            super().__init__(filename)
            # Dummy data for testing
            self._level_count = 3
            self._downsamples = [1.0, 2.0, 4.0]
            self._spacings = [(0.5, 0.5), (1.0, 1.0), (2.0, 2.0)]
            self._shapes = ((1000, 1000), (500, 500), (250, 250))

        def read_region(self, coordinates, level, size):
            arr = np.zeros(size + (3,), dtype=np.uint8)
            return fim.Image.from_numpy(arr)

        @property
        def properties(self):
            return {}

        @property
        def magnification(self):
            return 10.0

        @property
        def vendor(self):
            return "TestVendor"

        def close(self):
            pass

    def test_dummy_slide_backend_properties(self):
        slide = self.DummySlideBackend("test_filename.tiff")

        # Testing the level_count
        assert slide.level_count == 3

        # Testing the dimensions
        assert slide.dimensions == (1000, 1000)

        # Testing the spacing
        assert slide.spacing == (0.5, 0.5)

        # Testing the level_dimensions
        assert slide.level_dimensions == ((1000, 1000), (500, 500), (250, 250))

        # Testing the level_spacings
        assert slide.level_spacings == ((0.5, 0.5), (1.0, 1.0), (2.0, 2.0))

        # Testing the level_downsamples
        assert slide.level_downsamples == (1.0, 2.0, 4.0)

        # Testing slide bounds
        assert slide.slide_bounds == ((0, 0), (1000, 1000))

        # Testing get_best_level_for_downsample
        assert slide.get_best_level_for_downsample(0.5) == 0
        assert slide.get_best_level_for_downsample(1.0) == 0
        assert slide.get_best_level_for_downsample(2.0) == 1
        assert slide.get_best_level_for_downsample(3.0) == 1
        assert slide.get_best_level_for_downsample(4.5) == 2

    def test_repr(self):
        slide = self.DummySlideBackend("test_filename.tiff")
        assert (
            repr(slide) == "<DummySlideBackend(filename=test_filename.tiff, "
            "dimensions=(1000, 1000), spacing=(0.5, 0.5), "
            "magnification=10.0, "
            "vendor=TestVendor, "
            "level_count=3, "
            "mode=None)>"
        )

    def test_spacing_without_set(self):
        slide = self.DummySlideBackend("test_filename.tiff")
        slide._spacings = None
        assert slide.spacing is None

    def test_get_thumbnail(self):
        slide = self.DummySlideBackend("test_filename.tiff")

        # Getting a 200x200 thumbnail
        thumbnail = slide.get_thumbnail(200)
        assert isinstance(thumbnail, fim.Image)
        dims = thumbnail.dimensions
        assert dims[0] == 200 and dims[1] == 200  # width, height
        assert dims[2] == 3  # channels

        # Getting a 300x150 thumbnail
        thumbnail = slide.get_thumbnail((300, 150))
        assert isinstance(thumbnail, fim.Image)
        dims = thumbnail.dimensions
        # The aspect ratio should be preserved, so width might be less than 300
        assert dims[1] == 150  # height
        assert dims[2] == 3  # channels
