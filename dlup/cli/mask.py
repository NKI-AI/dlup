# coding=utf-8
# Copyright (c) dlup contributors
"""CLI utilities to handle masks"""
import argparse
import json
from functools import partial
from multiprocessing import Pool
from typing import cast

import numpy as np
import shapely
from shapely.ops import unary_union
from tqdm import tqdm

from dlup._image import Resampling
from dlup.annotations import AnnotationType, Polygon, WsiAnnotations, WsiSingleLabelAnnotation
from dlup.cli import file_path
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.experimental_backends import ImageBackend
from dlup.tiling import TilingMode
from dlup.utils.mask import mask_to_polygons


def _get_sample(index, dataset, index_map, scaling):
    output = {}
    sample = dataset[index]
    _mask = np.asarray(sample["image"])
    for index in index_map:
        curr_mask = (_mask == index).astype(np.uint8)
        if curr_mask.sum() == 0:
            continue
        output[index_map[index]] = mask_to_polygons(curr_mask, offset=sample["coordinates"], scaling=scaling)
    return output


def dataset_to_polygon(dataset, index_map, num_workers=0, scaling=1.0, show_progress=True):
    output_polygons: dict[str, list[Polygon]] = {v: [] for v in index_map.values()}

    sample_function = partial(_get_sample, dataset=dataset, index_map=index_map, scaling=scaling)

    if num_workers <= 0:
        for idx in tqdm(range(len(dataset)), disable=not show_progress):
            curr_polygons = sample_function(idx)
            for polygon_name in output_polygons:
                if polygon_name in curr_polygons:
                    output_polygons[polygon_name] += curr_polygons[polygon_name]
    else:
        with Pool(num_workers) as pool:
            with tqdm(total=len(dataset), disable=not show_progress) as pbar:
                for curr_polygons in pool.imap(sample_function, range(len(dataset))):
                    pbar.update()
                    for polygon_name in output_polygons:
                        if polygon_name in curr_polygons:
                            output_polygons[polygon_name] += curr_polygons[polygon_name]

    geometry = {k: unary_union(polygons) for k, polygons in output_polygons.items()}
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

    # Parse the labels
    if args.labels is None:
        index_map = {1: "label"}
    else:
        index_map = {}
        for pair in args.labels.split(","):
            name, index = pair.split("=")
            if not index.isnumeric():
                raise argparse.ArgumentTypeError(f"Expected a key-pair of the form 1=tumor,2=stroma")
            index = float(index)
            if not index.is_integer():
                raise argparse.ArgumentTypeError(f"Expected a key-pair of the form 1=tumor,2=stroma")
            index = int(index)
            if index == 0:
                raise argparse.ArgumentTypeError(f"0 is not a proper index. Needs to be at least 1.")
            index_map[index] = name.strip()

    polygons = dataset_to_polygon(dataset, index_map=index_map, num_workers=args.num_workers, scaling=scaling)
    wsi_annotations = []
    for label in polygons:
        if polygons[label].isempty:
            continue

        if isinstance(polygons[label], shapely.geometry.multipolygon.MultiPolygon):
            coordinates = [Polygon(coords, label=label) for coords in polygons[label].geoms if not coords.is_empty]
        else:
            coordinates = [Polygon(polygons[label], label=label)]

        wsi_annotations.append(
            WsiSingleLabelAnnotation(
                label=label,
                type=AnnotationType.POLYGON,
                coordinates=coordinates,
            )
        )

    slide_annotations = WsiAnnotations(wsi_annotations)
    if args.simplify is not None:
        slide_annotations.simplify(tolerance=args.simplify)

    if not args.separate:
        with open(output_filename, "w") as f:
            json.dump(slide_annotations.as_geojson(split_per_label=False), f, indent=2)
    else:
        jsons = slide_annotations.as_geojson(split_per_label=True)
        for label, json_dict in jsons:
            suffix = output_filename.suffix
            name = output_filename.with_suffix("").name
            new_name = name + "-" + label
            new_filename = (output_filename.parent / new_name).with_suffix(suffix)
            with open(new_filename, "w") as f:
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
        help="Filename of the mask. If `--separate` is set, will create a label <MASK_FN>-<label>.json",
    )
    mask_parser.add_argument(
        "OUTPUT_FN",
        type=lambda x: file_path(x, need_exists=False),
        help="Output filename. Will create all parent directories if needed.",
    )
    mask_parser.add_argument(
        "--labels",
        type=str,
        help="Comma-separated integer value in the mask, name. E.g. specimen=1,tumor=2,...",
    )
    mask_parser.add_argument(
        "--silent",
        action="store_true",
        help="If set, will not show progress bar.",
    )
    mask_parser.add_argument(
        "--separate",
        action="store_true",
        help="If set, save labels separately.",
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
        "--simplify",
        type=float,
        help="The maximum allowed geometry displacement. "
        "The higher this value, the smaller the number of vertices in the resulting geometry. "
        "By default this is disabled",
    )
    mask_parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of parallel threads to run. When 0 multiprocessing is disabled.",
    )

    mask_parser.set_defaults(subcommand=mask_to_polygon)
