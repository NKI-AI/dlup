# coding=utf-8
# Copyright (c) DLUP Contributors
import functools
import pathlib

from joblib import Parallel, delayed

from dlup.preprocessors.preprocessor import BackgroundMaskFunc, BasePreprocessor
from dlup.preprocessors.iterators import TileIterator
from dlup.slide import Slide
from dlup.utils.types import PathLike


class WsiBasePreprocessor(BasePreprocessor):
    def __init__(
        self,
        file_extension=".svs",
        border_mode: str = "crop",
        *args,
        **kwargs,
    ):
        super().__init__(
            *args,
            **kwargs,
        )

        self.file_extension = file_extension

        self.border_mode = border_mode
        self.requested_tile_size = self.tile_size

    def save_tile(self, save_dir: PathLike, tile_tuple) -> None:
        save_dir = pathlib.Path(save_dir)
        tile = tile_tuple.tile.convert("RGB")
        tile_save_path = (
            save_dir / "tiles" / f"tile_{'_'.join(map(str, tile_tuple.region.idx.tolist()))}{self.extension}"
        )
        tile.save(tile_save_path, quality=self.quality)

    def process_slide(self, slide: Slide) -> None:
        # TODO: Progress count
        # Reset tile count
        self.tile_count = 0
        self.background_tiles = 0

        # get current case ID and its index in the current pipeline
        slide_uid = slide.slide_uid
        patient_uid = slide.patient_uid

        # Directory where the tiles of the current tile and its metadata will be saved
        save_dir = self.output_dir / patient_uid / slide_uid
        # Create folders for metadata and tiles.
        (save_dir / self.metadata_dir).mkdir(exist_ok=True, parents=True)
        (save_dir / self.json_dir).mkdir(exist_ok=True, parents=True)
        (save_dir / self.tiles_dir).mkdir(exist_ok=True, parents=True)

        # Print properties of the slide we're currently working on
        self.logger.info(slide)

        if not self.filter_background:
            background_mask_func = None
        else:

            # TODO: Mask can also be used for first selecting all the regions in the tissue (labelled or not)
            # Currently not implemented (should have little effect)
            background_mask_func = functools.partial(BackgroundMaskFunc, background_threshold=self.background_threshold)

        tile_iterator = TileIterator(
            slide=slide,
            region_left=0,
            region_top=0,
            region_width=slide.width,
            region_height=slide.height,
            magnification=self.magnification,
            mpp=self.mpp,
            tile_size=self.tile_size,
            tile_overlap=self.tile_overlap,
            border_mode=self.border_mode,
            background_mask_func=background_mask_func,
        )

        num_tiles = tile_iterator.num_tiles
        self.logger.info(f"Will iterate over {num_tiles} tiles in the image.")

        # Save a thumbnail of the processed slide
        # TODO: Add grid lines
        # TODO: This requires that we know which tiles were rejected!
        # TODO: Cannot access tissue mask from the partial!
        # self.save_thumbnail(slide, save_dir, background_mask_func.tissue_mask)

        process_tile_partial = functools.partial(self.process_tile, save_dir)
        if self.num_workers > 0:
            Parallel(n_jobs=self.num_workers, backend="threading")(
                delayed(process_tile_partial)(tile_tuple) for tile_tuple in tile_iterator
            )
        else:
            for tile_tuple in tile_iterator:
                process_tile_partial(tile_tuple)

        total_processed = self.tile_count + self.background_tiles
        if total_processed != num_tiles:
            raise ValueError(
                f"Final index should be equal to the total number processed. Got {num_tiles} and {total_processed}."
            )

        task_str = (
            f"Processed {num_tiles} tiles for slide {slide.slide_uid}:\n"
            f" - Saved {self.tile_count} tiles.\n"
            f" - Dismissed {self.background_tiles} background tiles "
        )

        self._save_preprocessing_metadata(save_dir, slide_uid, tile_iterator, n_tiles=num_tiles)

        self.logger.info(task_str)
