import json
import time
from pathlib import Path

import cv2 as cv2

from dlup.annotations import WsiAnnotations
from dlup.annotations_experimental import WsiAnnotationsExperimental as WsiAnnotations2
from dlup.data.transforms import convert_annotations
from dlup.annotations_experimental import convert_annotations as convert_annotations_new

fn = Path("TCGA-E9-A1R4-01Z-00-DX1.B04D5A22-8CE5-49FD-8510-14444F46894D.geojson")
import numpy as np

start_time = time.time()
annotations = WsiAnnotations.from_geojson(fn, sorting="NONE")
import PIL.Image

print(f"Time to load annotations (dlup v0.7.0): {(time.time() - start_time):.5f}s")


# Bounding box:
bbox = annotations.bounding_box
print(f"Bounding box: {bbox}")
region_start = (500, 0)

# Let's get the region
start_time = time.time()
region = annotations.read_region(region_start, 0.02, bbox[1])
dlup_reg = time.time() - start_time
print(f"Time to read region (dlup v0.7.0): {dlup_reg:.5f}s")
print(f"Number of polygons in region (dlup v0.7.0): {len(region)}")
print()

start_time = time.time()
annotations2 = WsiAnnotations2.from_geojson(fn)
print(f"Time to load annotations (dlup v0.8.0.beta): {(time.time() - start_time):.5f}s")

start_time = time.time()
region2 = annotations2.read_region(region_start, 0.02, bbox[1])

new_reg = time.time() - start_time
print(f"Time to read (dlup v0.8.0.beta): {new_reg:.5f}s")
# print(f"Number of polygons in region (dlup v0.8.0.beta): {len(region2)}")

with open("dlup_region.json", "w") as f:
    json.dump(annotations2.as_geojson(), f, indent=2)

# Let's get all label names
labels0 = set([_.label for _ in region])
# print(f"Labels 1: {labels}")
#

print(f"Factor faster: {((dlup_reg / new_reg)):.3f}")
print(type(annotations2._layers.polygons[0]))
print(type(region2.polygons[0]))

labels1 = set([_.label for _ in (region2.polygons + region2.points)])

assert labels0 == labels1

# print(annotations2._layers.polygons[:2])
# print(annotations2._layers.polygons[0].label)


index_map = {
    "tissue (area)": 1,
    "artefact air bubble (area)": 2,
    "artefact mechanical expansion (area)": 3,
    "artefact mechanical compression (area)": 4,
    "artefact out of focus (area)": 5,
    "artefact pen marking (area)": 6,
}

for ann in annotations2._layers.polygons:
    assert ann.index is None

annotations2.reindex_polygons(index_map)

for ann in annotations2._layers.polygons:
    assert index_map[ann.label] == ann.index

color_map = {}
for r in region:
    if r.label not in color_map:
        color_map[index_map[r.label]] = r.color
LUT = np.zeros((6 + 1, 3), dtype=np.uint8)
for key, color in color_map.items():
    LUT[key] = color

print(LUT)

np.asarray((56630.2124, 69640.6535)) * 0.02
region_size = (1393, 1133)

_, mask, _ = convert_annotations(region, region_size=region_size, index_map=index_map)
print(mask.shape)


PIL.Image.fromarray(LUT[mask]).resize((1133 // 2, 1393 // 2)).save("dlup_original.png")


mask3 = convert_annotations_new(region2.polygons, region_size=region_size, index_map=index_map)
PIL.Image.fromarray(LUT[mask3]).resize((1133 // 2, 1393 // 2)).save("dlup_new.png")
