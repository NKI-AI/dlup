from dlup.fast_tiff_writer import FastTiffWriter
from dlup.writers import ImageWriter

class FastTifffileImageWriter(ImageWriter):
    def __init__(self, file_path, size, mpp, compression, tile_size, quality, pyramid):
        self._writer = FastTiffWriter(
            file_path,
            size,
            mpp,
            tile_size,
            compression,
            quality,
            pyramid,
        )

    def from_tiles_iterator(self, tiles_iterator):
        for tile in tiles_iterator:
            self._writer.write_tile(tile)
