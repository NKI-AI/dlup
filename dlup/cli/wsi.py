# Copyright (c) dlup contributors
import argparse
import json
import pathlib
from multiprocessing import Pool
from pathlib import Path
from typing import Any, cast

from PIL import Image

from dlup import SlideImage
from dlup.background_deprecated import AvailableMaskFunctions, get_mask  # type: ignore
from dlup.data.dataset import TiledWsiDataset
from dlup.tiling import TilingMode
from dlup.utils import ArrayEncoder
from dlup.viz.plotting import plot_2d


def tiling(args: argparse.Namespace) -> None:
    """Perform the WSI tiling."""
    input_file_path = args.slide_file_path
    output_directory_path = args.output_directory_path
    tile_size = cast(tuple[int, int], (args.tile_size,) * 2)
    tile_overlap = cast(tuple[int, int], (args.tile_overlap,) * 2)

    image = SlideImage.from_file_path(input_file_path)
    mask = get_mask(slide=image, mask_func=AvailableMaskFunctions[args.mask_func])

    # the nparray and PIL.Image.size height and width order are flipped is as it would be as a PIL.Image.
    # Below [::-1] casts the thumbnail_size to the PIL.Image expected size
    thumbnail_size = cast(tuple[int, int], mask.shape[::-1])
    thumbnail = image.get_thumbnail(thumbnail_size)

    # Prepare output directory.
    output_directory_path.mkdir(parents=True, exist_ok=True)

    Image.fromarray(mask.astype(dtype=bool)).save(output_directory_path / "mask.png")
    plot_2d(thumbnail).save(output_directory_path / "thumbnail.png")
    plot_2d(thumbnail, mask=mask).save(output_directory_path / "thumbnail_with_mask.png")

    # TODO: Maybe give the SlideImageDataset an image as input?
    dataset = TiledWsiDataset.from_standard_tiling(
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
        for grid_local_coordinates, idx in pool.imap(tile_saver.save_tile, range(num_tiles)):
            indices[idx] = grid_local_coordinates

    output["output"]["num_tiles"] = num_tiles
    output["output"]["tile_indices"] = indices
    output["output"]["background_tiles"] = len(dataset.regions) - num_tiles

    with open(output_directory_path / "tiles.json", "w") as file:
        json.dump(output, file, indent=2, cls=ArrayEncoder)


class TileSaver:
    def __init__(self, dataset: TiledWsiDataset, output_directory_path: Path, do_not_save_tiles: bool = False) -> None:
        self.dataset = dataset
        self.output_directory_path = output_directory_path
        self.do_not_save_tiles = do_not_save_tiles

    # TODO: Fix the Any
    def save_tile(self, index: int) -> tuple[Any, int]:
        tile_dict = self.dataset[index]
        tile = tile_dict["image"]
        grid_local_coordinates = tile_dict["grid_local_coordinates"]
        grid_index = tile_dict["grid_index"]

        if len(self.dataset.grids) > 1:
            indices = (grid_index,) + grid_local_coordinates

        if not self.do_not_save_tiles:
            tile.save(self.output_directory_path / f"{'_'.join(map(str, indices))}.png")

        return grid_local_coordinates, index


def info(args: argparse.Namespace) -> None:
    """Return available slide properties."""
    slide = SlideImage.from_file_path(args.slide_file_path)
    props = slide.properties
    if not props:
        return print("No properties found.")
    if args.json:
        print(json.dumps(dict(props)))
        return

    for k, v in props.items():
        print(f"{k}\t{v}")


def register_parser(parser: argparse._SubParsersAction) -> None:  # type: ignore
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
        "--num-workers",
        type=int,
        help="Number of parallel threads to run. None -> fully parallelized.",
    )
    tiling_parser.add_argument(
        "--do-not-save-tiles",
        dest="do_not_save_tiles",
        action="store_true",
        help="Flag to show what would have been tiled. If set -> saves metadata and masks, but does not perform tiling",
    )
    tiling_parser.set_defaults(do_not_save_tiles=False)

    tiling_parser.add_argument(
        "--mask-func",
        dest="mask_func",
        type=str,
        default="improved_fesi",
        choices=AvailableMaskFunctions.__members__,
        help="Function to compute the tissue mask with",
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
