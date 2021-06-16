# coding=utf-8
# Copyright (c) dlup contributors

import argparse
import json
import pathlib
from typing import Tuple, cast

import PIL

from dlup import SlideImage, SlideImageTiledRegionView
from dlup.background import foreground_tiles_coordinates_mask, get_mask
from dlup.tiling import TilingMode
from dlup.utils import ArrayEncoder


def tiling(args: argparse.Namespace):
    """Perform some tiling."""
    input_file_path = args.slide_file_path
    output_directory_path = args.output_directory_path
    tile_size = cast(Tuple[int, int], (args.tile_size,) * 2)
    tile_overlap = cast(Tuple[int, int], (args.tile_overlap,) * 2)

    image = SlideImage.from_file_path(input_file_path)
    scaled_view = image.get_scaled_view(image.mpp / args.mpp)
    tiled_view = SlideImageTiledRegionView(scaled_view, tile_size, tile_overlap, args.mode, crop=args.crop)

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
            "mpp": scaled_view.mpp,
            "size": scaled_view.size,
            "coordinates_grid_shape": tiled_view.coordinates_grid.shape,
        },
        "settings": {
            "mode": args.mode,
            "crop": args.crop,
            "tile_size": args.tile_size,
            "tile_overlap": args.tile_overlap,
            "mask_function": "fesi",
            "foreground_threshold": args.foreground_threshold,
        },
    }
    # Prepare output directory.
    output_directory_path /= "tiles"
    columns = tiled_view.coordinates_grid.shape[1]

    # Generate the foreground mask
    mask = get_mask(image)
    boolean_mask = foreground_tiles_coordinates_mask(mask, tiled_view, args.foreground_threshold)

    added_coords = []
    # Iterate through the tiles and save them in the provided location.
    for i, (tile, coords) in filter(lambda x: boolean_mask[x[0]], enumerate(zip(tiled_view, tiled_view.coordinates))):
        pil_image = PIL.Image.fromarray(tile)
        row = i // columns
        column = i % columns
        added_coords.append((row, column))
        output_directory_path.mkdir(parents=True, exist_ok=True)
        pil_image.save(output_directory_path / f"{row}_{column}.png")

    output["output"]["num_tiles"] = i + 1
    output["output"]["tile_coordinates"] = added_coords
    output["output"]["background_tiles"] = len(tiled_view.coordinates) - i - 1

    with open(output_directory_path / "tiles.json", "w") as file:
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
        help="Input directory.",
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
