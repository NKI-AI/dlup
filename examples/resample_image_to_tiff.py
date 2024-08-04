# Copyright (c) dlup contributors
"""
Resample a WSI to a TIFF file. This script resamples a WSI to a TIFF file with a given mpp and tile size.

This script shows the dlup functionality to read a WSI file in tiles and write this tile-by-tile to a tiff file.

"""

import argparse
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt

from dlup import SlideImage
from dlup.data.dataset import TiledWsiDataset
from dlup.writers import LibtiffImageWriter, TiffCompression, TifffileImageWriter


def resample(args: argparse.Namespace) -> None:
    # tile_size is the size of the tiles in the dataset, and coincides with the tile sizes in the TIFF writer.
    tile_size = args.tile_size
    tile_overlap = (0, 0)
    mpp = args.mpp

    dataset = TiledWsiDataset.from_standard_tiling(
        args.input,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        mpp=mpp,
        crop=False,
        internal_handler="vips",
        backend="PYVIPS",
    )
    scaled_region_view = dataset.slide_image.get_scaled_view(dataset.slide_image.get_scaling(mpp))

    writer_class = LibtiffImageWriter if args.use_libtiff else TifffileImageWriter

    writer = writer_class(
        args.output,
        size=(*scaled_region_view.size, 3),
        mpp=(mpp, mpp),
        compression=TiffCompression.JPEG,
        tile_size=tile_size[::-1],
        quality=85,
        pyramid=True,
    )

    def tiles_iterator(dataset: TiledWsiDataset) -> Iterator[npt.NDArray[np.int_]]:
        for tile in dataset:
            # TODO: Convert VIPS image to RGB directly
            arr = tile["image"].flatten(background=(255, 255, 255)).numpy()
            yield arr

    start = time.time()
    writer.from_tiles_iterator(tiles_iterator(dataset))
    end = time.time()
    print(f"Time to write the TIFF file: {end - start:.2f} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample a WSI to a TIFF file.")
    parser.add_argument("input", type=Path, help="Path to the WSI file.")
    parser.add_argument("output", type=Path, help="Path to the output TIFF file.")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(512, 512), help="Size of the tiles in the TIFF.")
    parser.add_argument("--mpp", type=float, required=False, help="Microns per pixel of the output TIFF file.")
    parser.add_argument(
        "--use-libtiff",
        action="store_true",
        help="Use libtiff for writing the TIFF file, otherwise use a tifffile.py writer.",
    )
    args = parser.parse_args()

    with SlideImage.from_file_path(args.input, internal_handler="vips", backend="PYVIPS") as img:
        print("Image information: %s" % img)
        native_mpp = img.mpp

    if not args.mpp:
        args.mpp = native_mpp

    print(
        "Resampling image %s to TIFF with mpp %s and internal tile size %s and saving to %s"
        % (args.input, args.mpp, args.tile_size, args.output)
    )

    resample(args)


if __name__ == "__main__":
    main()
