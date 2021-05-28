# coding=utf-8
# Copyright (c) DLUP Contributors
import abc
import functools
import logging
import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import PIL
from numpy.typing import ArrayLike

from dlup.preprocessors.background import get_mask
from dlup.preprocessors.iterators import Region, TileIterator
from dlup.preprocessors.slide_manager import BaseSlideManager
from dlup.slide import Slide
from dlup.utils.bbox import crop_to_bbox
from dlup.utils.io import read_json, write_json
from dlup.utils.types import PathLike
from dlup.viz.plotting import plot_2d


@dataclass
class MaskTile:
    mask: np.ndarray
    bbox: list


class BackgroundMask:
    def __init__(self, slide: Slide, level: int):
        self.slide = slide

        # Compute the scaling factor between the mask and the current level
        mask_shape = np.asarray(self.tissue_mask.shape)[::-1]
        # This is the scaling between the mask shape # TODO: Why does it return level 0 so often?
        self.scaling = mask_shape / slide.levels[level].shape

    @functools.cached_property
    def tissue_mask(self) -> np.ndarray:
        return get_mask(self.slide)

    def __getitem__(self, region: Region) -> MaskTile:
        # Coordinates need to be scaled to the closest level
        coordinates = region.coordinates * self.slide.levels[region.level].downsample * self.scaling
        size = region.size * self.scaling
        bbox = [
            *np.floor(coordinates).astype(int).tolist()[::-1],
            *np.ceil(size).astype(int).tolist()[::-1],
        ]
        # We check the mask in the region *before* interpolation
        return MaskTile(mask=crop_to_bbox(self.tissue_mask, bbox=bbox), bbox=bbox)


class BasePreprocessor(abc.ABC):
    def __init__(
        self,
        output_dir: PathLike,
        filter_background: bool = True,
        background_threshold: float = 0.05,
        quality: int = 95,
        mpp: Optional[bool] = None,
        magnification: Optional[bool] = None,
        tile_size: int = 512,
        tile_overlap: int = 56,
        compression: str = "png",
        num_workers: int = 0,
        *args,
        **kwargs,
    ):
        self.logger = logging.getLogger(type(self).__name__)

        # Where to save
        self.output_dir = pathlib.Path(output_dir)
        self.filter_background = filter_background
        self.background_threshold = background_threshold

        # Output settings
        self.mpp = mpp
        self.magnification = magnification
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.tile_count = 0

        # Output formats.
        self.quality = quality
        self.compression = compression
        self.metadata_manager = None  # To fix.

        # These are the directory names to save the tile metadata and tiles themselves to.
        self.metadata_dir = "metadata"
        self.tiles_dir = "tiles"
        self.json_dir = "json"

        self.num_workers = num_workers

        self.background_tiles = 0
        self.tile_count = 0

    @property
    def extension(self) -> str:
        if self.compression == "png":
            return ".png"
        self.logger.error(f"Compression {self.compression} not supported.")
        raise RuntimeError

    @staticmethod
    def _save_progress(save_dir, tile_count, slide_completed=False) -> None:
        write_json(
            save_dir / "progress.json",
            {"tile_count": tile_count, "done": slide_completed},
        )

    @staticmethod
    def _get_progress(save_dir: pathlib.Path) -> Tuple[int, bool]:
        data = read_json(save_dir / "progress.json")
        return data["tile_count"], data["done"]

    def save_thumbnail(self, slide: Slide, save_dir: pathlib.Path, background_mask: bool = None):
        """
        Save a small thumbnail for the current slide.
        """
        # TODO: Add a grid on top of the thumbnail!
        # TODO: Add discarded tiles on top of the thumbnail (preferable two thumbnails)
        thumbnail = slide.get_thumbnail().convert("RGB")
        thumbnail.save(save_dir / ("thumbnail" + self.extension))

        if background_mask is not None:
            mask = PIL.Image.fromarray(np.uint8(background_mask * 255), "L")
            mask.save(save_dir / ("tissue_mask" + self.extension))
            mask = mask.resize(np.asarray(thumbnail).shape[:-1][::-1])
            thumbnail_with_contours = plot_2d(np.asarray(thumbnail), mask=np.asarray(mask))
            thumbnail_with_contours.save(save_dir / ("thumbnail_mask" + self.extension))

    def _save_tile_metadata(
        self,
        tile_tuple,
        save_dir,
    ):
        filename_no_ext = f"tile_{'_'.join(map(str, tile_tuple.region.idx.tolist()))}"
        tile_metadata = {**tile_tuple.region, "tile_filename": f"{filename_no_ext}{self.extension}"}
        write_json(save_dir / self.json_dir / filename_no_ext + ".json", tile_metadata)

    def _save_preprocessing_metadata(self, save_dir, slide_uid, tile_iterator, n_tiles):
        # TODO: Need to add a bit more of the iterator info itself
        slide_metadata = {
            "slide_uid": slide_uid,
            "n_tiles": n_tiles,
            "n_saved_tiles": self.tile_count,
            "n_background_tiles": self.background_tiles,
            "quality": self.quality,
            "iterator_info": {k: v for k, v in tile_iterator.iter_info.items() if k != "indices"},
            **{
                k: v for k, v in tile_iterator.__dict__.items() if k not in ["slide", "iter_info"]
            },  # TODO: iter_info needs to be added.
            "slide": {
                k: v
                for k, v in tile_iterator.slide.__dict__.items()
                if k
                in [
                    "shape",
                    "is_lossy_tiff",
                    "mpp_x",
                    "mpp_y",
                    "n_channels",
                    "patient_uid",
                    "slide_uid",
                    "properties",
                ]
            },
            "file_path": str(tile_iterator.slide.file_name),
        }

        # TODO: Also add metadata
        write_json(save_dir / (slide_uid + ".json"), slide_metadata)

    @abc.abstractmethod
    def save_tile(self, save_dir: PathLike, tile_tuple) -> None:
        pass

    @abc.abstractmethod
    def process_slide(self, slide: Slide):
        pass

    def process_tile(self, background_mask: np.ndarray, save_dir: PathLike, tile_tuple: ArrayLike):
        self.logger.debug(f"Working on tile {tile_tuple.region.idx}...")
        # If tissue mask, then we can check for background
        if self.filter_background:
            curr_mask = background_mask[tile_tuple.region]
            is_foreground = curr_mask.mask.mean() >= self.background_threshold
            self.logger.debug(f"{tile_tuple.region.idx} is foreground: {is_foreground}")
            if not is_foreground:
                self.background_tiles += 1
                return

        self.save_tile(save_dir, tile_tuple)
        self._save_tile_metadata(tile_tuple, save_dir)
        self.tile_count += 1

    def run(self, slide_manager: BaseSlideManager):
        for slide_idx, slide in enumerate(slide_manager):
            self.logger.info(f"Processing slide {slide_idx + 1} of {len(slide_manager)}.")
            self.process_slide(slide)
            slide.close()
