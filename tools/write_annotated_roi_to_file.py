# coding=utf-8
# Copyright (c) dlup Contributors
"""
This module writes annotated ROIs to disk. It creates for each ROI a grid with one tile precisely the size of this ROI.
To use for your setting, likely you need to change `mask_color_map` and `geometries_color_map`.
"""
import argparse
import pathlib

import numpy as np

from dlup import SlideImage
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.data.transforms import ConvertAnnotationsToMask
from dlup.experimental_annotations import WsiAnnotations
from dlup.experimental_backends import ImageBackends
from dlup.tiling import Grid, TilingMode
from dlup.viz.plotting import plot_2d


def _create_roi_grid(rois, scaling, requested_mpp):
    """
    Helper function to create grids which each precisely cover one ROI.

    Parameters
    ----------
    rois
    scaling
    requested_mpp

    Returns
    -------

    """
    grids = []
    for offset, size in rois:
        offset = tuple(np.ceil(np.asarray(offset) * scaling).astype(int))
        size = tuple(np.floor(np.asarray(size) * scaling).astype(int))
        grid = Grid.from_tiling(
            offset,
            size=size,
            tile_size=size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
        )
        grids.append((grid, size, requested_mpp))
    return grids


def create_roi_dataset(image_fn, annotations_dir):
    image_fn = pathlib.Path(image_fn)
    annotations_dir = pathlib.Path(annotations_dir)
    annotations = annotations_dir.glob("*.json")

    annotations = WsiAnnotations.from_geojson([annotations_dir / _ for _ in annotations])

    rois = annotations["roi"].bounding_boxes

    backend = ImageBackends.PYVIPS
    with SlideImage.from_file_path(image_fn, backend=backend) as slide_image:
        mpp = slide_image.mpp

    requested_mpp = 0.5
    scaling = mpp / requested_mpp
    # Create the grids
    grids = _create_roi_grid(rois, scaling, requested_mpp)

    mask_map = {"stroma": 1, "tumor": 2, "inflamed": 3, "mild-inflamed": 4}

    transform = ConvertAnnotationsToMask("roi", index_map=mask_map)
    dataset = TiledROIsSlideImageDataset(
        image_fn,
        grids=grids,
        crop=False,
        mask=None,
        annotations=annotations,
        transform=transform,
        backend=backend,
    )

    return dataset


def save_dataset(output_dir, ds, mask_color_map, geometry_color_map):

    for idx, sample in enumerate(ds):
        tile = sample["image"]
        mask = sample["annotation_data"]["mask"]
        image = plot_2d(
            tile,
            mask,
            mask_colors=mask_color_map,
            mask_alpha=40,
            geometries=sample["annotations"],
            geometries_color_map=geometry_color_map,
        )
        image.save(output_dir / f"image_roi_{idx}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("IMAGE_PATH", type=pathlib.Path)
    parser.add_argument("ANNOT_DIR", type=pathlib.Path)
    parser.add_argument("OUTPUT_DIR", type=pathlib.Path)

    args = parser.parse_args()
    dataset = create_roi_dataset(args.IMAGE_PATH, args.ANNOT_DIR)
    mask_color_map = {1: "darkgreen", 2: "pink", 3: "red", 4: "lightgreen"}
    geometries_color_map = {
        "plasmacells": "darkblue",
        "tumorcells": "blue",
        "redbloodcells": "orange",
        "lymphocytes": "red",
    }

    save_dataset(args.OUTPUT_DIR, dataset, mask_color_map=mask_color_map, geometry_color_map=geometries_color_map)
