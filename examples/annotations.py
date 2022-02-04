import dlup
from shapely.geometry import mapping
from dlup import SlideImage
from dlup.data._annotations import SlideAnnotations
import rasterio.features
import numpy as np
from dlup.viz.plotting import plot_2d
from dlup.tiling import Grid
import PIL
import pathlib
from dlup import BoundaryMode
from dlup.data.dataset import ConcatDataset

from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup import SlideImage
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from dlup.background import get_mask
from dlup.tiling import TilingMode
import errno
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


import logging
logger = logging.getLogger()

def construct_dataset_per_wsi(slide_fn, annotation_fn, tile_size, target_mpp=None):
    image = SlideImage.from_file_path(slide_fn)
    if not target_mpp:
        target_mpp = image.mpp

    scaling = image.mpp / target_mpp

    annotations = SlideAnnotations.from_asap_xml(annotation_fn)
    # TIGER stores the ROIs in the 'roi' key
    bboxes = annotations["roi"].bounding_boxes(scaling)

    grids = []
    for bbox in bboxes:
        offset = bbox[:2]
        size = bbox[2:]

        # We CEIL the offset and FLOOR the size, so that we are always in a fully annotated area.
        offset = np.ceil(offset).astype(int)
        size = np.floor(size).astype(int)

        curr_grid = Grid.from_tiling(
            offset,
            size=size,
            tile_size=tile_size,
            tile_overlap=(0, 0),
            mode=TilingMode.overflow,
        )

        grids.append((curr_grid, tile_size, target_mpp))

    dataset = TiledROIsSlideImageDataset(slide_fn, grids, annotations=annotations, crop=False)
    return dataset


def build_dataset(path_to_images, path_to_annotations, tile_size, target_mpp):
    # Let us find all the images
    annotations = pathlib.Path(path_to_annotations).glob("*.xml")
    path_to_images = pathlib.Path(path_to_images)
    per_image_dataset = []
    failures = []

    for xml_file in tqdm(annotations):
        if "122S.xml" not in str(xml_file.name):
            continue
        logger.info(f"Adding {xml_file}")
        # Now we can get the path as follows:
        image_file = (path_to_images / xml_file.name).with_suffix(".tif")
        if not image_file.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(image))

        per_image_dataset.append(construct_dataset_per_wsi(
            image_file, xml_file, tile_size=tile_size, target_mpp=target_mpp))
    return ConcatDataset(per_image_dataset), failures


def visualize_tile(d, label_map):
    # TODO: MultiPolyGon does not work, and preferably we do not want borders after cropping in the polygon
    size = tuple(d["image"].shape[:2])
    output = Image.new("RGBA", size, (255, 255, 255, 255))
    tile = PIL.Image.fromarray(d["image"])
    coords = np.array(d["coordinates"])
    draw = ImageDraw.Draw(output)
    output.paste(tile, (0, 0) + size)
    for label, label_color in label_map.items():
        label_name = f"label_{label}"
        if label_name in d and d[label_name]:
            for curr_polygon in d[label_name]:
                if curr_polygon.type == "MultiPolygon":
                    continue
                try:
                    polygons = [np.asarray(curr_polygon.exterior.coords)]
                    for polygon in polygons:
                        draw.polygon([item for sublist in polygon for item in sublist], outline=label_color, width=3)
                except:
                    continue

    return output


if __name__ == "__main__":
    tile_size = (1024, 1024)
    dataset, failures = build_dataset(
        "/mnt/archive/data/pathology/TIGER/tiger-training-data/wsirois/wsi-level-annotations/images/",
        "/mnt/archive/data/pathology/TIGER/tiger-training-data/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/",
        tile_size, None)

    label = "label_tumor-associated stroma"
    # for d in tqdm(dataset):
    #     labels = d[label]
    #     image = d["image"]
    #     mask = np.zeros(image.shape[:2], dtype=np.uint8)
    #     for label in labels:
    #         mask += rasterio.features.rasterize([label], out_shape=image.shape[:2])
    #         print()
    #     mask_img = mask * 255
    #     img = np.asarray(plot_2d(image, mask=mask))
    #     print()

    label_map = {
        "healthy glands": "red",
        # "invasive tumor": "black",
        #     "rest": "purple",
        "tumor-associated stroma": "green",
        "lymphocytes and plasma cells": "orange",
    #     "roi": "yellow",
    }
    sample = dataset[8]
    print(f"Keys: {sample.keys()}")
    data = np.asarray(visualize_tile(sample, label_map))
    print()
