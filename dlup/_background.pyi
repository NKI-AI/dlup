# Copyright (c) dlup contributors
from typing import Any, List

import numpy as np
import numpy.typing as npt

from dlup import SlideImage

def _is_foreground_numpy(
    slide_image: SlideImage,
    background_mask: npt.NDArray[np.int_],
    regions: List[Any],  # TODO: Specialize
    boolean_mask: npt.NDArray[np.bool_],
    threshold: float = 1.0,
) -> None: ...
