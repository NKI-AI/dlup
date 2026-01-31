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
"""Utilities for handling WSIs."""

import math

from dlup._exceptions import UnsupportedSlideError


def check_if_mpp_is_valid(mpp_x: float, mpp_y: float, *, rel_tol: float = 0.015) -> None:
    """
    Checks if the mpp is (nearly) isotropic and defined. The maximum allowed rel_tol

    Parameters
    ----------
    mpp_x : float
    mpp_y : float
    rel_tol : float
        Relative tolerance between mpp_x and mpp_y. 1.5% seems to work well for most slides.

    Returns
    -------
    None
    """
    if mpp_x == 0 or mpp_y == 0:
        raise UnsupportedSlideError("Unable to parse mpp.")

    if not mpp_x or not mpp_y or not math.isclose(mpp_x, mpp_y, rel_tol=rel_tol):
        raise UnsupportedSlideError(f"cannot deal with slides having anisotropic mpps. Got {mpp_x} and {mpp_y}.")
