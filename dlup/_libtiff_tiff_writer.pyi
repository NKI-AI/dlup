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
