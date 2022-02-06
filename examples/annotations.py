import errno
import logging
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import rasterio.features
from PIL import Image, ImageDraw
from shapely.geometry import box, mapping
from tqdm import tqdm

import dlup
from dlup import BoundaryMode, SlideImage
from dlup.background import get_mask
from dlup.data._annotations import AnnotationType, SlideAnnotations
from dlup.data.dataset import ConcatDataset, TiledROIsSlideImageDataset
from dlup.tiling import Grid, TilingMode
from dlup.viz.plotting import plot_2d

logger = logging.getLogger()


# invasive tumor (label=1): this class contains regions of the invasive tumor, including several morphological subtypes, such as invasive ductal carcinoma and invasive lobular carcinoma;
# tumor-associated stroma (label=2): this class contains regions of stroma (i.e., connective tissue) that are associated with the tumor. This means stromal regions contained within the main bulk of the tumor and in its close surrounding; in some cases, the tumor-associated stroma might resemble the "healthy" stroma, typically found outside of the tumor bulk;
# in-situ tumor (label=3): this class contains regions of in-situ malignant lesions, such as ductal carcinoma in situ (DCIS) or lobular carcinoma in situ (LCIS).
# healthy glands (label=4): this class contains regions of glands with healthy epithelial cells;
# necrosis not in-situ (label=5): this class contains regions of necrotic tissue that are not part of in-situ tumor; for example, ductal carcinoma in situ (DCIS) often presents a typical necrotic pattern, which can be considered as part of the lesion itself, such a necrotic region is not annotated as "necrosis" but as "in-situ tumor";
# inflamed stroma (label=6): this class contains tumor-associated stroma that has a high density of lymphocytes (i.e., it is "inflamed"). When it comes to assessing the TILs, inflamed stroma and tumor-associated stroma can be considered together, but were annotated separately to take into account for differences in their visual patterns;
# rest (label=7): this class contains regions of several tissue compartments that are not specifically annotated in the other categories; examples are healthy stroma, erythrocytes, adipose tissue, skin, nipple, etc.


def _cast_polygon(x, _):
    return x, AnnotationType.POLYGON


def _cast_as_box(x, _):
    return box(*x.bounds), AnnotationType.BOX


def _convert_box_to_point(x, _):
    return x.centroid, AnnotationType.POINT


TIGER_LABEL_CONVERSIONS = {
    "healthy glands": _cast_polygon,
    "in-situ tumor": _cast_polygon,
    "inflamed stroma": _cast_polygon,
    "invasive tumor": _cast_polygon,
    "tumor-associated stroma": _cast_polygon,
    "lymphocytes and plasma cells": _convert_box_to_point,  #
    "necrosis not in-situ": _cast_polygon,
    "rest": _cast_polygon,
    "roi": _cast_as_box,
}

all_labels = list(TIGER_LABEL_CONVERSIONS.keys())


def construct_dataset_per_wsi(slide_fn, annotation_fn, tile_size, target_mpp=None):
    image = SlideImage.from_file_path(slide_fn)
    if not target_mpp:
        target_mpp = image.mpp

    scaling = image.mpp / target_mpp

    annotations = SlideAnnotations.from_asap_xml(annotation_fn, label_map=TIGER_LABEL_CONVERSIONS)
    # TIGER stores the ROIs in the 'roi' key
    bboxes = annotations["roi"].bounding_boxes(scaling)

    # Now we can remove roi from the available labels, as we are not interested anymore.
    annotations.available_labels = [_ for _ in annotations.available_labels if _ != "roi"]

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


def build_dataset(path_to_images, path_to_annotations, tile_size, target_mpp, transform=None):
    # Let us find all the images
    xml_files = pathlib.Path(path_to_annotations).glob("*.xml")
    path_to_images = pathlib.Path(path_to_images)
    per_image_dataset = []
    for xml_filename in tqdm(xml_files):
        logger.info(f"Adding {xml_filename}")
        image_filename = (path_to_images / xml_filename.name).with_suffix(".tif")
        if not image_filename.exists():
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(image_filename))

        per_image_dataset.append(
            construct_dataset_per_wsi(image_filename, xml_filename, tile_size=tile_size, target_mpp=target_mpp, transform=transform)
        )
    return ConcatDataset(per_image_dataset)


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
    dataset = build_dataset(
        "/mnt/archive/data/pathology/TIGER/tiger-training-data/wsirois/wsi-level-annotations/images/",
        "/mnt/archive/data/pathology/TIGER/tiger-training-data/wsirois/wsi-level-annotations/annotations-tissue-cells-xmls/",
        tile_size,
        None,
    )

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
