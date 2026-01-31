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

import numpy as np
import numpy.typing as npt

def get_foreground_indices_numpy(
    image_width: int,
    image_height: int,
    image_slide_average_mpp: float,
    background_mask: npt.NDArray[np.int_],
    regions_array: npt.NDArray[np.float64],
    threshold: float,
    foreground_indices: npt.NDArray[np.int64],
) -> int: ...
