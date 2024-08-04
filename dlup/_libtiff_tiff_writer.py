# Copyright (c) dlup contributors
"""This module is only required for the linters"""
from typing import Any

import numpy as np
from numpy.typing import NDArray


class LibtiffTiffWriter:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def write_tile(self, tile: NDArray[np.int_], row: int, col: int) -> None:
        pass

    def write_pyramid(self) -> None:
        pass

    def finalize(self) -> None:
        pass
