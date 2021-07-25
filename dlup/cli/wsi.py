# coding=utf-8
# Copyright (c) dlup contributors

import argparse
import json
import pathlib
from typing import Tuple, cast

import numpy as np
import PIL

from dlup import SlideImage
from dlup.background import get_mask
from dlup.data.dataset import SlideImageDataset
from dlup.tiling import TilingMode
from dlup.utils import ArrayEncoder
from dlup.viz.plotting import plot_2d


def tiling(args: argparse.Namespace):
    """Perform the WSI tiling."""
    input_file_path = args.slide_file_path
    output_directory_path = args.output_directory_path
    tile_size = cast(Tuple[int, int], (args.tile_size,) * 2)
    tile_overlap = cast(Tuple[int, int], (args.tile_overlap,) * 2)

    image = SlideImage.from_file_path(input_file_path)
    mask = get_mask(image)

    thumbnail_size = cast(Tuple[int, int], mask.shape[::-1])
    thumbnail = image.get_thumbnail(thumbnail_size)

    plot_2d(mask).save(output_directory_path / "mask.png")
    plot_2d(thumbnail).save(output_directory_path / "thumbnail.png")
    plot_2d(thumbnail, mask=mask).save(output_directory_path / "thumbnail_with_mask.png")

    # TODO: Maybe give the SlideImageDataset an image as input?
    dataset = SlideImageDataset(
        input_file_path,
        mask=mask,
        mpp=args.mpp,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        tile_mode=args.mode,
        foreground_threshold=args.foreground_threshold,
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
            "mpp": dataset.mpp,
            "size": dataset.region_view.size,
        },
        "settings": {
            "mode": args.mode,
            "crop": args.crop,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "mask_function": "improved_fesi",
            "foreground_threshold": args.foreground_threshold,
        },
    }
    # Prepare output directory.
    output_directory_path /= "tiles"

    indices = []
    # Iterate through the tiles and save them in the provided location.
    for idx in range(num_tiles):
        tile_dict = dataset[idx]
        tile = PIL.Image.fromarray(tile_dict["image"])
        grid_index = tile_dict["grid_index"]

        indices.append(grid_index)
        output_directory_path.mkdir(parents=True, exist_ok=True)
        tile.save(output_directory_path / f"{'_'.join(map(str, grid_index))}.png")

    output["output"]["num_tiles"] = num_tiles
    output["output"]["tile_indices"] = indices
    output["output"]["background_tiles"] = len(dataset.grid) - num_tiles

    with open(output_directory_path.parent / "tiles.json", "w") as file:
        json.dump(output, file, indent=2, cls=ArrayEncoder)


def info(args: argparse.Namespace):
    """Return available slide properties."""
    slide = SlideImage.from_file_path(args.slide_file_path)
    props = slide.properties
    if args.json:
        print(json.dumps(dict(props)))
        return

    for k, v in props.items():
        print(f"{k}\t{v}")


def register_parser(parser: argparse._SubParsersAction):
    """Register wsi commands to a root parser."""
    wsi_parser = parser.add_parser("wsi", help="WSI parser")
    wsi_subparsers = wsi_parser.add_subparsers(help="WSI subparser")
    wsi_subparsers.required = True
    wsi_subparsers.dest = "subcommand"

    # Tile a slide and save the tiles in an output folder.
    tiling_parser = wsi_subparsers.add_parser("tile", help="Generate tiles in a target output folder.")
    tiling_parser.add_argument(
        "--tile-size",
        type=int,
        required=True,
        help="Size of the generated tiles.",
    )
    tiling_parser.add_argument(
        "--tile-overlap",
        type=int,
        default=0,
        help="Number of overlapping pixels between tiles.",
    )
    tiling_parser.add_argument(
        "--mode",
        type=TilingMode,
        default=TilingMode.skip,
        choices=TilingMode.__members__,
        help="Policy to handle verflowing tiles.",
    )
    tiling_parser.add_argument(
        "--crop",
        action="store_true",
        help="If a tile is overflowing, crop it.",
    )
    tiling_parser.add_argument(
        "--foreground-threshold",
        type=float,
        default=0.0,
        help="Fraction of foreground to consider a tile valid.",
    )
    tiling_parser.add_argument(
        "--mpp",
        type=float,
        required=True,
        help="Microns per pixel.",
    )
    tiling_parser.add_argument(
        "slide_file_path",
        type=pathlib.Path,
        help="Input slide image.",
    )
    tiling_parser.add_argument(
        "output_directory_path",
        type=pathlib.Path,
        help="Directory to save output too.",
    )
    tiling_parser.set_defaults(subcommand=tiling)

    # Get generic slide infos.
    tiling_parser = wsi_subparsers.add_parser("info", help="Return available slide properties.")
    tiling_parser.add_argument(
        "slide_file_path",
        type=pathlib.Path,
        help="Input slide image.",
    )
    tiling_parser.add_argument(
        "--json",
        action="store_true",
        help="Print available properties in json format.",
    )
    tiling_parser.set_defaults(subcommand=info)
