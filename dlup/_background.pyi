# Copyright (c) dlup contributors

import numpy as np
import numpy.typing as npt

def _get_foreground_indices_numpy(
    image_width: int,
    image_height: int,
    image_slide_average_mpp: float,
    background_mask: npt.NDArray[np.int_],
    regions_array: npt.NDArray[np.float64],
    threshold: float,
    foreground_indices: npt.NDArray[np.int64],
) -> int: ...
