from dlup import SlideImage
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import TilingMode
from dlup.writers import TiffImageWriter, TiffCompression
import numpy as np

target_mpp = 11.4
tile_size = [1024, 1024]

compress_fn = "/processing/j.teuwen/compressed_file.tiff"

dataset_temp = TiledROIsSlideImageDataset.from_standard_tiling(
    compress_fn, target_mpp, tile_size, (0, 0), mask=None, tile_mode=TilingMode.overflow
)
scaling = dataset_temp.slide_image.get_scaling(target_mpp)
scaling = 1.0
image_size_temp = dataset_temp.slide_image.get_scaled_size(scaling)

temp_slide = SlideImage.from_file_path(compress_fn)

print()
