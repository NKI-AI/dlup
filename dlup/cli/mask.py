# coding=utf-8
# Copyright (c) dlup contributors
"""CLI utilities to handle masks"""
import argparse
import json
from multiprocessing import Pool
from typing import cast

import numpy as np
import shapely
from shapely.ops import unary_union
from tqdm import tqdm

from dlup._image import Resampling
from dlup.cli import file_path
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.experimental_backends import ImageBackend
from dlup.tiling import TilingMode
from dlup.utils.mask import generate_polygons


def dataset_to_polygon(dataset, num_workers=0, scaling=1.0, show_progress=True):
    polygons = []

    def get_sample(idx):
        sample = dataset[idx]
        _mask = np.asarray(sample["image"])
        if _mask.sum() == 0:
            return None
        return generate_polygons(_mask, offset=sample["coordinates"], scaling=scaling)

    if num_workers <= 0:
        for idx in tqdm(range(len(dataset)), disable=not show_progress):
            polygon = get_sample(idx)
            if polygon is not None:
                polygons.append(polygon)
    else:
        with Pool(num_workers) as pool:
            with tqdm(total=len(dataset), disable=not show_progress) as pbar:
                for polygon in pool.imap(get_sample, range(len(dataset))):
                    pbar.update()
                    if polygon is not None:
                        polygons.append(polygon)

    geometry = unary_union(polygons)
    return geometry


def mask_to_polygon(args: argparse.Namespace):
    """Perform the mask conversion to polygon."""
    mask_filename = args.MASK_FN
    output_filename = args.OUTPUT_FN
    tile_size = cast(tuple[int, int], (args.tile_size,) * 2)
    tile_overlap = cast(tuple[int, int], (args.tile_overlap,) * 2)

    # Prepare output directory.
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    dataset = TiledROIsSlideImageDataset.from_standard_tiling(
        mask_filename,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        mpp=None,
        tile_mode=TilingMode.overflow,
        crop=False,
        backend=ImageBackend.TIFFFILE,
        interpolator=Resampling.NEAREST,
    )
    target_mpp = args.mpp
    if target_mpp is None:
        scaling = 1.0
    else:
        scaling = dataset.slide_image.get_scaling(target_mpp=target_mpp)

    polygon = dataset_to_polygon(dataset, num_workers=args.num_workers, scaling=scaling)
    json_dict = shapely.geometry.mapping(polygon)

    with open(output_filename, "w") as f:
        json.dump(json_dict, f, indent=2)


def register_parser(parser: argparse._SubParsersAction):
    """Register mask commands to a root parser."""
    wsi_parser = parser.add_parser("mask", help="WSI mask parser")
    wsi_subparsers = wsi_parser.add_subparsers(help="WSI mask subparser")
    wsi_subparsers.required = True
    wsi_subparsers.dest = "subcommand"

    # Tile a slide and save the tiles in an output folder.
    mask_parser = wsi_subparsers.add_parser(
        "mask-to-polygon", help="Convert a WSI defining a mask to a GeoJSON polygon."
    )
    mask_parser.add_argument(
        "MASK_FN",
        type=file_path,
        help="Filename of the mask.",
    )
    mask_parser.add_argument(
        "OUTPUT_FN",
        type=lambda x: file_path(x, need_exists=False),
        help="Output filename. Will create all parent directories if needed.",
    )
    # mask_parser.add_argument(
    #     "--labels",
    #     type=str,
    #     help="Comma-separated integer value in the mask, name. E.g. specimen=1,tumor=2,...",
    # )
    mask_parser.add_argument(
        "--silent",
        action="store_true",
        help="If set, will not show progress bar.",
    )
    mask_parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Size of the generated tiles.",
    )
    mask_parser.add_argument(
        "--tile-overlap",
        type=int,
        default=128,
        help="Number of overlapping pixels between tiles.",
    )
    mask_parser.add_argument(
        "--interpolation",
        type=Resampling,
        default=Resampling.NEAREST,
        choices=Resampling.__members__,
        help="Policy to handle interpolation.",
    )
    mask_parser.add_argument(
        "--mpp",
        type=float,
        required=False,
        help="Microns per pixel the resulting polygon should be converted to, e.g.,"
        " to rescale to level 0 of and image. If not set, will pick the mask mpp (so no scaling).",
    )
    mask_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel threads to run. When 0 multiprocessing is disabled.",
    )

    mask_parser.set_defaults(subcommand=mask_to_polygon)
