import pathlib
from typing import Tuple
from dlup.tiling import TilingMode
from dlup.tiling import TiledRegionView
from dlup import SlideImage

from torch.utils.data import Dataset


class SlideImageDataset(Dataset, TiledRegionView):
    """Basic Slide Image dataset."""

    def __init__(self, path: pathlib.Path, mpp: float, tile_size: Tuple[int, int],
                 tile_overlap: Tuple[int, int], background_threshold: float = 0.0):
        slide_image = SlideImage.from_file_path(path)
        scaled_view = slide_image.get_scaled_view(slide_image.mpp / mpp)
        super().__init__(scaled_view, tile_size, tile_overlap, TilingMode.overflow, crop=True)
        # coordinates_background_mask = compute_background_mask(mask, self.coordinates)
        # background_mask =

    def __getitem__(self, i):
        return TiledRegionView.__getitem__(self, i)

    def __len__(self):
        return TiledRegionView.__len__(self)
