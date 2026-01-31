# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
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
import warnings

from dlup.utils.imports import OPENSLIDE_AVAILABLE, TIFFFILE_AVAILABLE

if not OPENSLIDE_AVAILABLE:
    warnings.warn(
        "Openslide is not available. OpenSlide backend will not be available. To install it, run `pip install openslide-python`."
    )
else:
    from .openslide_backend import OpenSlideSlide as OpenSlideSlide  # noqa: F401

if not TIFFFILE_AVAILABLE:
    warnings.warn(
        "Tifffile is not available. Tifffile backend will not be available. To install it, run `pip install tifffile`."
    )
else:
    from .tifffile_backend import TifffileSlide as TifffileSlide  # noqa: F401

from .fastslide_backend import FastSlideSlide as FastSlideSlide  # noqa: F401
