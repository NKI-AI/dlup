# coding=utf-8
# Copyright (c) dlup contributors

import argparse
import json
import pathlib
import sys
from multiprocessing import Pool
from typing import Tuple, cast

import numpy as np
from PIL import Image

from dlup import SlideImage
from dlup._cache import create_tiff_cache
from dlup.background import AvailableMaskFunctions, get_mask
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.tiling import Grid, TilingMode
from dlup.utils import ArrayEncoder
from dlup.viz.plotting import plot_2d
from dlup.writers import TiffCompression, TiffImageWriter


def tiling(args: argparse.Namespace):
    """Perform the WSI tiling."""
    input_file_path = args.slide_file_path
    output_directory_path = args.output_directory_path
    tile_size = cast(Tuple[int, int], (args.tile_size,) * 2)
    tile_overlap = cast(Tuple[int, int], (args.tile_overlap,) * 2)

    image = SlideImage.from_file_path(input_file_path)

    mask_func = AvailableMaskFunctions[args.mask_func]
    mask = get_mask(slide=image, mask_func=mask_func) if mask_func != mask_func.none else None

    # the nparray and PIL.Image.size height and width order are flipped is as it would be as a PIL.Image.
    # Below [::-1] casts the thumbnail_size to the PIL.Image expected size
    thumbnail_size = cast(Tuple[int, int], mask.shape[::-1])
    thumbnail = image.get_thumbnail(thumbnail_size)

    # Prepare output directory.
    output_directory_path.mkdir(parents=True, exist_ok=True)

    if args.mask_format == "png":
        Image.fromarray(mask.astype(dtype=bool)).save(output_directory_path / "tissue_mask.png")
    else:
        mpp = image.mpp * np.asarray(image.size) / mask.shape[::-1]
        # TODO: Figure out why bit_depth 8 is really needed here.
        writer = TiffImageWriter(
            mpp=mpp, size=mask.size, compression=TiffCompression.CCITTFAX4, bit_depth=8, silent=True
        )
        # TODO: Why is 255 needed? x255
        writer.from_pil(Image.fromarray(mask * 255), output_directory_path / "tissue_mask.tiff")

    plot_2d(thumbnail).save(output_directory_path / "thumbnail.png")
    plot_2d(thumbnail, mask=mask).save(output_directory_path / "thumbnail_with_mask.png")

    # TODO: Maybe give the SlideImageDataset an image as input?
    dataset = TiledROIsSlideImageDataset.from_standard_tiling(
        input_file_path,
        mask=mask,
        mpp=args.mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_mode=args.mode,
        mask_threshold=args.foreground_threshold,
        transform=None,
    )
    num_tiles = len(dataset)

    # Store metadata of this process for reproducibility (and to read in the dataset class)
    output = {
        "original": {
            "input_file_path": str(input_file_path),
            "aspect_ratio": image.aspect_ratio,
            "magnification": image.magnification,
            "mpp": image.mpp,
            "size": image.size,
            "vendor": image.vendor,
        },
        "output": {
            "mpp": args.mpp,
            "size": dataset.slide_image.get_scaled_size(dataset.slide_image.get_scaling(args.mpp)),
        },
        "settings": {
            "mode": args.mode,
            "crop": args.crop,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "mask_function": args.mask_func,
            "foreground_threshold": args.foreground_threshold,
        },
    }

    indices = [None for _ in range(num_tiles)]

    # Iterate through the tiles (and save them in the provided location)
    tiles_output_directory_path = output_directory_path / "tiles"
    if not args.do_not_save_tiles:
        tiles_output_directory_path.mkdir(parents=True, exist_ok=True)
    tile_saver = TileSaver(dataset, tiles_output_directory_path, do_not_save_tiles=args.do_not_save_tiles)

    with Pool(args.num_workers) as pool:
        for (grid_local_coordinates, idx) in pool.imap(tile_saver.save_tile, range(num_tiles)):
            indices[idx] = grid_local_coordinates

    output["output"]["num_tiles"] = num_tiles
    output["output"]["tile_indices"] = indices
    output["output"]["background_tiles"] = len(dataset.regions) - num_tiles

    with open(output_directory_path / "tiles.json", "w") as file:
        json.dump(output, file, indent=2, cls=ArrayEncoder)


class TileSaver:
    def __init__(self, dataset, output_directory_path, do_not_save_tiles=False):
        self.dataset = dataset
        self.output_directory_path = output_directory_path
        self.do_not_save_tiles = do_not_save_tiles

    def save_tile(self, index):
        tile_dict = self.dataset[index]
        tile = tile_dict["image"]
        grid_local_coordinates = tile_dict["grid_local_coordinates"]
        grid_index = tile_dict["grid_index"]

        indices = grid_local_coordinates
        if len(self.dataset.grids) > 1:
            indices = [grid_index] + indices

        if not self.do_not_save_tiles:
            tile.save(self.output_directory_path / f"{'_'.join(map(str, indices))}.png")

        return grid_local_coordinates, index


def info(args: argparse.Namespace):
    """Return available slide properties."""
    slide = SlideImage.from_file_path(args.slide_file_path)
    props = slide.properties
    if args.json:
        print(json.dumps(dict(props)))
        return

    for k, v in props.items():
        print(f"{k}\t{v}")


def downsample(args: argparse.Namespace):
    """Downsample a WSI"""
    suffix = args.output_slide_file_path.suffixes[-1]
    if suffix not in [".tif", ".tiff"]:
        sys.exit(f"Can only downsample to TIFF images. Received {suffix}.")

    slide_image = SlideImage.from_file_path(args.slide_file_path)
    if slide_image.mpp > args.mpp:
        sys.exit(f"Input image has mpp {slide_image.mpp}, while requested mpp is {args.mpp}. Exiting.")
    slide_level_size = slide_image.get_scaled_size(slide_image.get_scaling(args.mpp))
    grid = Grid.from_tiling(
        (0, 0),
        size=slide_level_size,
        tile_size=(args.tile_size, args.tile_size),
        tile_overlap=(0, 0),
        mode=TilingMode.overflow,
    )

    create_tiff_cache(
        slide_image=slide_image,
        grid=grid,
        mpp=args.mpp,
        output_size=slide_level_size,
        tile_size=(args.tile_size, args.tile_size),
        filename=args.output_slide_file_path,
        tiff_tile_size=(args.tile_size, args.tile_size),
        pyramid=args.pyramid,
    )


def register_parser(parser: argparse._SubParsersAction):
    """Register wsi commands to a root parser."""
    wsi_parser = parser.add_parser("wsi", help="WSI parser")
    wsi_subparsers = wsi_parser.add_subparsers(help="WSI subparser")
    wsi_subparsers.required = True
    wsi_subparsers.dest = "subcommand"

    # Tile a slide and save the tiles in an output folder.
    wsi_parser = wsi_subparsers.add_parser("tile", help="Generate tiles in a target output folder.")
    wsi_parser.add_argument(
        "--tile-size",
        type=int,
        required=True,
        help="Size of the generated tiles.",
    )
    wsi_parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Number of overlapping pixels between tiles.",
    )
    wsi_parser.add_argument(
        "--mode",
        type=TilingMode,
        default=TilingMode.skip,
        choices=TilingMode.__members__,
        help="Policy to handle overflowing tiles.",
    )
    wsi_parser.add_argument(
        "--crop",
        action="store_true",
        help="If a tile is overflowing, crop it.",
    )
    wsi_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.0,
        help="Fraction of foreground to consider a tile valid.",
    )
    wsi_parser.add_argument(
        "--mpp",
        type=float,
        required=True,
        help="Microns per pixel.",
    )
    wsi_parser.add_argument(
        "--num-workers",
        type=int,
        help="Number of parallel threads to run. None -> fully parallelized.",
    )
    wsi_parser.add_argument(
        "--do-not-save-tiles",
        dest="do_not_save_tiles",
        action="store_true",
        help="Flag to show what would have been tiled. If set -> saves metadata and masks, but does not perform tiling",
    )
    wsi_parser.set_defaults(do_not_save_tiles=False)

    wsi_parser.add_argument(
        "--mask-func",
        dest="mask_func",
        type=str,
        default="improved_fesi",
        choices=AvailableMaskFunctions.__members__,
        help="Function to compute the tissue mask with",
    )
    wsi_parser.add_argument(
        "--mask-format",
        dest="mask_format",
        type=str,
        default="tiff",
        choices=["tiff", "png"],
        help="Write mask as a tiff or png",
    )

    wsi_parser.add_argument(
        "slide_file_path",
        type=pathlib.Path,
        help="Input slide image.",
    )
    wsi_parser.add_argument(
        "output_directory_path",
        type=pathlib.Path,
        help="Directory to save output too.",
    )
    wsi_parser.set_defaults(subcommand=tiling)

    # Get generic slide infos.
    wsi_parser = wsi_subparsers.add_parser("info", help="Return available slide properties.")
    wsi_parser.add_argument(
        "slide_file_path",
        type=pathlib.Path,
        help="Input slide image.",
    )
    wsi_parser.add_argument(
        "--json",
        action="store_true",
        help="Print available properties in json format.",
    )
    wsi_parser.set_defaults(subcommand=info)

    wsi_parser = wsi_subparsers.add_parser("downsample", help="Downsample a slide.")
    wsi_parser.add_argument(
        "slide_file_path",
        type=pathlib.Path,
        help="Input slide image.",
    )
    wsi_parser.add_argument(
        "output_slide_file_path",
        type=pathlib.Path,
        help="Output slide image. Only .tif or .tiff is allowed as an extension.",
    )
    wsi_parser.add_argument(
        "--mpp",
        type=float,
        required=True,
        help="Microns per pixel of the output image.",
    )
    wsi_parser.add_argument(
        "--tile-size",
        type=int,
        required=True,
        help="Size of the tile size used to write. "
        "Likely this has very small visual effect, nevertheless it might make sense to use the same tile size"
        "as you would when reading. Will also set the internal tiff tile size to this value.",
    )
    wsi_parser.add_argument(
        "--pyramid",
        action="store_true",
        help="Store as a pyramidal format. Is useful when the image will be further downsampled.",
    )
    # TODO: Compression, quality.

    wsi_parser.set_defaults(subcommand=downsample)
