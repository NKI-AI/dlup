import json
import time
from pathlib import Path

import cv2 as cv2

from dlup.annotations import WsiAnnotations
from dlup.annotations_experimental import WsiAnnotationsExperimental as WsiAnnotations2
from dlup.data.transforms import convert_annotations

fn = Path("TCGA-E9-A1R4-01Z-00-DX1.B04D5A22-8CE5-49FD-8510-14444F46894D.geojson")
import numpy as np

start_time = time.time()
annotations = WsiAnnotations.from_geojson(fn, sorting="NONE")
import PIL.Image

print(f"Time to load annotations (dlup v0.7.0): {(time.time() - start_time):.5f}s")



# TODO: Temporary here
def convert_annotations_new(
    annotations,
    region_size: tuple[int, int],
    default_value: int = 0,
    index_map: dict[str, int] = None,
):
    mask = np.empty(region_size, dtype=np.int32)
    mask[:] = default_value
    for curr_annotation in annotations:
        holes_mask = None
        index_value = index_map[curr_annotation.label]
        original_values = None
        interiors = [(np.asarray(pi)).round().astype(np.int32) for pi in curr_annotation.get_interiors()]
        if interiors != []:
            original_values = mask.copy()
            holes_mask = np.zeros(region_size, dtype=np.int32)
            # Get a mask where the holes are
            cv2.fillPoly(holes_mask, interiors, [1])

        cv2.fillPoly(
            mask,
            [(np.asarray(curr_annotation.get_exterior())).round().astype(np.int32)],
            [index_value],
        )
        if interiors != []:
            # TODO: This is a bit hacky to ignore mypy here, but I don't know how to fix it.
            mask = np.where(holes_mask == 1, original_values, mask)  # type: ignore
    return mask



# Bounding box:
bbox = annotations.bounding_box
print(f"Bounding box: {bbox}")
region_start = (0, 0)


# Let's get the region
start_time = time.time()
region = annotations.read_region(region_start, 0.02, bbox[1])
dlup_reg = time.time() - start_time
print(f"Time to read region (dlup v0.7.0): {dlup_reg:.5f}s")
print(f"Number of polygons in region (dlup v0.7.0): {len(region)}")
print()

start_time = time.time()
annotations2 = WsiAnnotations2.from_geojson(fn)
bbox = annotations2._layers.bounding_box
print(f"Bounding box v0.8.0.beta: {bbox}")

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

mask_ = region2.to_mask(region_size, index_map, 0)
PIL.Image.fromarray(LUT[mask_]).resize((1133 // 2, 1393 // 2)).save("dlup_new_opencv.png")

mask3 = convert_annotations_new(region2.polygons, region_size=region_size, index_map=index_map)
PIL.Image.fromarray(LUT[mask3]).resize((1133 // 2, 1393 // 2)).save("dlup_new.png")

print()
# Let's time everything separately.
print("Benchmark\n=========")

annotations = WsiAnnotations.from_geojson(fn, sorting="NONE")
bbox = annotations.bounding_box

start_time = time.time()
region = annotations.read_region(region_start, 0.02, bbox[1])
print(f"Time to read region (dlup v0.7.0): {(time.time() - start_time) * 1000:.2f}ms")
start_time2 = time.time()
_, mask, _ = convert_annotations(region, region_size=region_size, index_map=index_map)
print(f"Time to convert annotations to mask (dlup v0.7.0): {(time.time() - start_time2) * 1000:.2f}ms")
total_time = (time.time() - start_time)
print(f"Total time to read region and convert to mask (dlup v0.7.0): {total_time * 1000:.2f}ms")
print()
annotations2 = WsiAnnotations2.from_geojson(fn)
start_time = time.time()
region2 = annotations2.read_region(region_start, 0.02, bbox[1])
print(f"Time to read region (dlup v0.8.0.beta): {(time.time() - start_time) * 1000:.2f}ms")
start_time2 = time.time()
mask_ = region2.to_mask(region_size, index_map, 0)
print(f"Time to convert annotations to mask (dlup v0.8.0.beta): {(time.time() - start_time2) * 1000:.2f}ms")
total_time2 = (time.time() - start_time)
print(f"Total time to read region and convert to mask (dlup v0.8.0.beta): {total_time2 * 1000:.2f}ms")

print(f"\nSpeedup: {total_time/total_time2:.3f} times")
