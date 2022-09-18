# coding=utf-8
# Copyright (c) dlup contributors
"""This file contains utilities to work with ROIs.
"""
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
from tqdm import tqdm

from dlup import SlideImage
from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.data.transforms import ConvertAnnotationsToMask
from dlup.experimental_backends import ImageBackend
from dlup.tiling import GridOrder, TilingMode
from dlup.viz.plotting import plot_2d
from dlup.writers import TifffileImageWriter

base_path = Path("/Users/jteuwen")
image_fn = base_path / "TCGA-A2-A0YE-01Z-00-DX1.8A2E3094-5755-42BC-969D-7F0A2ECA0F39.svs"

annotation_jsons = Path("/Users/jteuwen/annotations/867/j.teuwen@nki.nl/88233").glob("*.json")
annotations = WsiAnnotations.from_geojson(annotation_jsons)

labels = annotations.available_labels
print(labels)

roi_annotations = annotations.copy()

roi_annotations.filter(["roi"])

mpp = 2
tile_size = (128, 128)
tile_overlap = (0, 0)
offset = (0, 0)


# # This is required, because we essentially need to know the size of the image before proceeding getting the roi_mask
# # This can be avoided if there is a method to update the annotations/transform after creating a dataset
# # The dataset itself already opens the image, and it would be great if we could avoid opening it twice.
with SlideImage.from_file_path(image_fn) as slide_image:
    scaling = slide_image.get_scaling(mpp)
    scaled_region_view = slide_image.get_scaled_view(scaling)

transform = ConvertAnnotationsToMask(
    roi_name=None, index_map={"roi": 1, "inflamed": 2, "normal glands": 3, "stroma": 4, "tumor": 5}
)

dataset = TiledROIsSlideImageDataset.from_standard_tiling(
    image_fn,
    mpp,
    tile_size=tile_size,
    tile_overlap=tile_overlap,
    tile_mode=TilingMode.overflow,
    grid_order=GridOrder.C,
    crop=False,
    mask=roi_annotations,
    mask_threshold=0,
    rois=None,
    annotations=annotations,
    labels=None,
    transform=transform,
    backend=ImageBackend.PYVIPS,
)

# # 40764 without mask
print(f"Dataset length: {len(dataset)}.")

image = PIL.Image.new("RGBA", tuple(scaled_region_view.size), (255, 255, 255, 255))

for d in tqdm(dataset):
    tile = d["image"]
    coords = np.array(d["coordinates"])
    box = tuple(np.array((*coords, *(coords + tile_size))).astype(int))
    mask = d["annotation_data"]["mask"]

    tile = plot_2d(
        tile, mask=mask, mask_colors={1: "red", 2: "green", 3: "purple", 4: "orange", 5: "black"}, mask_alpha=30
    )
    image.paste(tile, box)
    draw = PIL.ImageDraw.Draw(image)
    draw.rectangle(box, outline="red")

#
# roi = PIL.Image.fromarray(roi_mask * 255, mode="L")
# roi.save("roi.png", quality=90)

# image.thumbnail((4096, 4096))
image.save("output.png", quality=90)


#
#
# def iter():
#     prev_grid_local_coordinates = 0
#     for d in tqdm(dataset):
#         tile = d["image"]
#         local_coordinates = np.prod(np.array(d["grid_local_coordinates"]))
#         if local_coordinates != prev_grid_local_coordinates + 1:
#             for index in range(0, local_coordinates - prev_grid_local_coordinates):
#                 prev_grid_local_coordinates += 1
#                 yield np.zeros((*tile_size, 3), dtype=np.uint8)
#
#         mask = d["annotation_data"]["mask"]
#
#         tile = plot_2d(tile, mask=mask, mask_colors={1: "red"}, mask_alpha=30)
#         draw = PIL.ImageDraw.Draw(tile)
#         draw.rectangle((0, 0, *tile_size), outline="red")
#
#         yield np.asarray(tile.convert("RGB"))
#
#
writer = TifffileImageWriter(
    "output.tiff", size=(*tuple(scaled_region_view.size), 3), mpp=mpp, tile_size=tile_size, pyramid=True
)


def iterator():
    for sample in dataset:
        image = np.asarray(sample["image"])
        yield image


# writer.from_tiles_iterator(iterator())
print()
