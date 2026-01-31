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
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

class LibtiffTiffWriter:
    def __init__(
        self,
        file_path: str | Path,
        size: tuple[int, int, int],
        mpp: tuple[float, float],
        tile_size: tuple[int, int],
        compression: str,
        quality: int,
    ) -> None: ...
    def write_tile(self, tile: NDArray[np.int_], row: int, col: int) -> None: ...
    def write_pyramid(self) -> None: ...
    def finalize(self) -> None: ...
