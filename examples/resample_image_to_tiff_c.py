# Copyright (c) dlup contributors
"""
Resample a WSI to a TIFF file. This script resamples a WSI to a TIFF file with a given mpp and tile size.

This script shows the dlup functionality to read a WSI file in tiles and write this tile-by-tile to a tiff file.
"""

import argparse
from pathlib import Path
from typing import Iterator

import numpy as np
import numpy.typing as npt

from dlup import SlideImage
from dlup.data.dataset import TiledWsiDataset
from dlup.writers import TiffCompression
from dlup.c_writers import FastTifffileImageWriter


def resample(args: argparse.Namespace) -> None:
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
    print("Initialized dataset.")
    writer = FastTifffileImageWriter(
        args.output,
        size=(*scaled_region_view.size, 3),
        mpp=mpp,
        compression="JPEG",
        tile_size=tile_size[::-1],
        quality=85,
        pyramid=True,
    )
    import time

    start_time = time.time()

    tiles_x = (scaled_region_view.size[1] + tile_size[1] - 1) // tile_size[1]
    for idx, tile in enumerate(dataset):
        arr = tile["image"].flatten(background=(255, 255, 255)).numpy()
        row = (idx // tiles_x) * tile_size[0]
        col = (idx % tiles_x) * tile_size[1]
        writer._writer.write_tile(arr, row, col)
    writer._writer.flush()
    writer._writer.write_pyramid()
    writer._writer.finalize()
    # writer._writer.write_downsampled_pyramid(1)
    # writer._writer.flush()
    # writer._writer.write_downsampled_pyramid(2)

    # writer._writer.write_pyramid(1)
    # writer._writer.flush()
    # writer._writer.write_pyramid(2)

    print("Time taken: %s" % (time.time() - start_time))


def main() -> None:
    parser = argparse.ArgumentParser(description="Resample a WSI to a TIFF file.")
    parser.add_argument("input", type=Path, help="Path to the WSI file.")
    parser.add_argument("output", type=Path, help="Path to the output TIFF file.")
    parser.add_argument("--tile-size", type=int, nargs=2, default=(512, 512), help="Size of the tiles in the TIFF.")
    parser.add_argument("--mpp", type=float, required=False, help="Microns per pixel of the output TIFF file.")
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
