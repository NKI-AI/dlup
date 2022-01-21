from pathlib import Path
from typing import List, Optional, Union

from shapely import geometry

# from shapely.strtree import STRtree
import shapely

from enum import Enum
from typing import NamedTuple

# from collections import Sequence
import numpy as np
import json
from dlup import SlideImage
from dlup import BoundaryMode, SlideImage
from dlup.tiling import Grid
import rasterio.features
from dlup.data._annotations import SlideAnnotations, AnnotationParser
from dlup.viz.plotting import plot_2d


if __name__ == "__main__":

    mpp = 11.4
    tile_size = (1024, 1024)

    slide_image = SlideImage.from_file_path(
        "/processing/j.teuwen/TCGA-5T-A9QA-01Z-00-DX1.B4212117-E0A7-4EF2-B324-8396042ACEC1.svs"
    )

    parser = AnnotationParser.from_geojson(["/processing/j.teuwen/specimen.json"], ["specimen"], slide_image.mpp)
    annotations = SlideAnnotations(parser=parser)  # .get_annotations_for_labels(["specimen"])

    scaled_region_size = slide_image.get_scaled_size(slide_image.get_scaling(mpp))
    grid = Grid.from_tiling((0, 0), size=scaled_region_size, tile_size=(1024, 1024), tile_overlap=(0, 0))

    coordinates = grid[1]

    # for coordinates in grid:
    region_size = tile_size
    scaling: float = slide_image.mpp / mpp
    region_view = slide_image.get_scaled_view(scaling)
    region_view.boundary_mode = BoundaryMode.crop

    tile = slide_image.read_region(coordinates, scaling, region_size)

    tile_annotations = annotations.get_region(coordinates, region_size, mpp)
    mask = rasterio.features.rasterize(tile_annotations["specimen"], out_shape=tile_size)

    arr = np.asarray(plot_2d(tile, mask=mask))

    print()
