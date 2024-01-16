# Copyright (c) dlup contributors
"""Utilities to work with binary masks"""
from functools import partial
from multiprocessing import Pool
from typing import Sequence

import cv2
import numpy as np
import numpy.typing as npt
import shapely
import shapely.affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from dlup.data.dataset import TiledWsiDataset


def _DFS(
    polygons: list[Polygon],
    contours: Sequence[npt.NDArray[np.generic]],
    hierarchy: npt.NDArray[np.generic],
    sibling_id: int,
    is_outer: bool,
    siblings: list[npt.NDArray[np.generic]] | None,
    offset: tuple[int | float, int | float] = (0, 0),
    scaling: float = 1.0,
) -> None:
    # Adapted FROM: https://gist.github.com/stefano-malacrino/7d429e5d12854b9e51b187170e812fa4
    while sibling_id != -1:
        contour = contours[sibling_id].squeeze(axis=1)
        if len(contour) >= 3:
            first_child_id = hierarchy[sibling_id][2]
            children: list[npt.NDArray[np.generic]] | None = [] if is_outer else None
            _DFS(polygons, contours, hierarchy, first_child_id, not is_outer, children)

            if is_outer:
                polygon = Polygon(contour, holes=children)
                if offset is not None and offset != (0, 0):
                    transformation_matrix = [
                        scaling,
                        0,
                        0,
                        scaling,
                        offset[0],
                        offset[1],
                    ]
                    polygon = shapely.affinity.affine_transform(polygon, transformation_matrix)

                polygons.append(polygon)
            else:
                assert siblings is not None  # This is for mypy
                siblings.append(contour)

        sibling_id = hierarchy[sibling_id][0]


def mask_to_polygons(
    mask: npt.NDArray[np.int_], offset: tuple[int | float, int | float] = (0, 0), scaling: float = 1.0
) -> list[Polygon]:
    # Adapted From: https://gist.github.com/stefano-malacrino/7d429e5d12854b9e51b187170e812fa4

    """Generates a list of Shapely polygons from the contours hierarchy returned by cv2.find_contours().
     The list of polygons is generated by performing a depth-first search on the contours hierarchy tree.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask
    offset : tuple, optional
        The offset for the polygon
    scaling : float, optional
        The scaling for the polygon

    Returns
    -------
    list
        The list of generated Shapely polygons
    """
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    hierarchy = hierarchy[0]
    polygons: list[Polygon] = []
    _DFS(polygons, contours, hierarchy, 0, True, [], offset=offset, scaling=scaling)

    return polygons


def _get_sample(
    index: int, dataset: TiledWsiDataset, index_map: dict[int, str], scaling: float
) -> dict[str, list[Polygon]]:
    output: dict[str, list[Polygon]] = {}
    sample = dataset[index]
    _mask = np.asarray(sample["image"])
    for index in index_map:
        curr_mask = (_mask == index).astype(np.uint8)
        if not curr_mask.any():
            continue
        output[index_map[index]] = mask_to_polygons(curr_mask, offset=sample["coordinates"], scaling=scaling)
    return output


# TODO: show_progress should be a function that can be e.g. `tqdm.tqdm`
def dataset_to_polygon(
    dataset: TiledWsiDataset,
    index_map: dict[int, str],
    num_workers: int = 0,
    scaling: float = 1.0,
    show_progress: bool = True,
) -> dict[str, Polygon]:
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
