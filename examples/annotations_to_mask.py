# Copyright (c) dlup contributors
"""This code provides an example of how to convert annotations to a mask."""
import json
from pathlib import Path

import numpy as np
import PIL.Image

from dlup.annotations_experimental import SlideAnnotations

d_fn = Path("TCGA-E9-A1R4-01Z-00-DX1.B04D5A22-8CE5-49FD-8510-14444F46894D.json")

Z_INDICES = {
    "tissue (area)": 0,
    "artefact mechanical expansion (area)": 1,
    "artefact out of focus (area)": 2,
    "artefact edge margin ink (area)": 3,
    "artefact mechanical compression (area)": 3,
    "artefact other (area)": 4,
    "artefact air bubble (area)": 5,
    "artefact foreign object (area)": 5,
    "artefact coverslip (area)": 6,
    "artefact pen marking (area)": 7,
}

index_map = {
    "tissue (area)": 1,
    "artefact air bubble (area)": 2,
    "artefact mechanical expansion (area)": 3,
    "artefact mechanical compression (area)": 4,
    "artefact out of focus (area)": 5,
    "artefact pen marking (area)": 6,
}
annotations = SlideAnnotations.from_darwin_json(d_fn, z_indices=Z_INDICES, sorting="Z_INDEX")
scaling = 0.02

bbox = annotations.bounding_box_at_scaling(scaling)
annotations.reindex_polygons(index_map)
region = annotations.read_region((0, 0), scaling, bbox[1])
LUT = annotations.color_lut
import time

start_time = time.time()
curr_mask = region.polygons_eager.to_mask()
print(f"Time to compute mask eagerly: {time.time() - start_time}")


print(region, "region")
print(region.polygons, "region.polygons")  # This should be lazy

# for polygon in region.polygons.get_geometries():
#     print(polygon)
polys = region.polygons.get_geometries()  # This should start computing

import time

start_time = time.time()
curr_mask = region.polygons.to_mask().numpy()
print(f"Time to compute mask lazily: {time.time() - start_time}")


print(curr_mask)
print(np.asarray(curr_mask).shape)

mask_itself = region.polygons.to_mask().numpy()

mask = LUT[mask_itself]

PIL.Image.fromarray(mask).save("mask.png")
from dlup.geometry import Box, GeometryCollection

collection = GeometryCollection()
polygon = Box((1, 1), (4, 4)).as_polygon()
polygon.index = 2
collection.add_polygon(polygon)

region = collection.read_region((0, 0), 1.0, (5, 5))

print("Python: Getting geometries")
# region.polygons.get_geometries()
print("Python: got geometries")
print("Python: Computing mask")
mask = np.asarray(region.polygons.to_mask())
print("Python: Got mask")
# print(mask)
# assert mask.sum() == 16 * 2
print("Python: Done")
mask = np.asarray(region.polygons.to_mask())


# print("Getting geometries")

# # for polygon in region.polygons.get_geometries():
# #     print(polygon)

# with open("test.xml", "w") as f:
#     f.write(annotations.as_dlup_xml())


# with open("test.geojson", "w") as f:
#     f.write(json.dumps(annotations.as_geojson(), indent=2))

# annotations2 = SlideAnnotations.from_dlup_xml("test.xml")
# region2 = annotations2.read_region((0, 0), scaling, bbox[1])
# LUT = annotations2.color_lut

# mask = LUT[region.polygons.to_mask().numpy()]
# PIL.Image.fromarray(mask).save("mask2.png")
