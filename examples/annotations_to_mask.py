# # Copyright (c) dlup contributors
# """This code provides an example of how to convert annotations to a mask."""

# from pathlib import Path

# import PIL.Image

# from dlup.annotations import WsiAnnotations
# from dlup.annotations_experimental import SlideAnnotations
# from dlup.data.transforms import convert_annotations
# from dlup.geometry import Point, Polygon, GeometryCollection

# # exterior = [(0, 0), (0, 3), (3, 3), (3, 0)]
# # interiors = [[(1.5, 1.5), (1.5, 2.5), (2.5, 2.5), (2.5, 1.5)]]
# # expected_area = 7.0


# # collection = GeometryCollection()
# # collection.add_polygon(DlupPolygon(exterior, interiors))

# # collection.add_point(DlupPoint(0, 0))

# # print(collection.polygons)
# # print(collection.points)


# # point = DlupPoint(1, 1)
# # point.scale(2)

# # print(point.get_coordinates())

# # polygon = DlupPolygon(exterior, interiors)

# # print(polygon.get_interiors(), polygon.get_exterior())
# # polygon.scale(2)
# # print(polygon.get_interiors(), polygon.get_exterior())

# # print(polygon.get_exterior)

# # print(collection.bounding_box)


# # collection = GeometryCollection()
# # collection.add_polygon(DlupPolygon(exterior, interiors))
# # collection.add_point(DlupPoint(1, 1))

# # print(collection.polygons[0].get_exterior())
# # print(collection.polygons[0].get_interiors())
# # print(collection.points[0].get_coordinates())
# # print(collection.bounding_box)
# # print("Scaling")
# # collection.scale(2.5)
# # print(collection.polygons[0].get_exterior())
# # print(collection.polygons[0].get_interiors())
# # print(collection.points[0].get_coordinates())
# # print(collection.bounding_box)

# # collection.scale(1 / 2.5)

# # print(collection.polygons[0].get_exterior())
# # print(collection.polygons[0].get_interiors())
# # print(collection.points[0].get_coordinates())
# # print(collection.bounding_box)

# # collection.set_offset((1, 2))

# # print(collection.polygons[0].get_exterior())
# # print(collection.polygons[0].get_interiors())
# # print(collection.points[0].get_coordinates())
# # print(collection.bounding_box)

# fn = Path("/Users/j.teuwen/Downloads/TCGA-E9-A1R4-01Z-00-DX1.B04D5A22-8CE5-49FD-8510-14444F46894D.geojson")
# d_fn = Path(
#     "/Users/j.teuwen/Downloads/v7_artifacts_v3.1/TCGA-E9-A1R4-01Z-00-DX1.B04D5A22-8CE5-49FD-8510-14444F46894D.json"
# )


# Z_INDICES = {
#     "tissue (area)": 0,
#     "artefact mechanical expansion (area)": 1,
#     "artefact out of focus (area)": 2,
#     "artefact edge margin ink (area)": 3,
#     "artefact mechanical compression (area)": 3,
#     "artefact other (area)": 4,
#     "artefact air bubble (area)": 5,
#     "artefact foreign object (area)": 5,
#     "artefact coverslip (area)": 6,
#     "artefact pen marking (area)": 7,
# }

# index_map = {
#     "tissue (area)": 1,
#     "artefact air bubble (area)": 2,
#     "artefact mechanical expansion (area)": 3,
#     "artefact mechanical compression (area)": 4,
#     "artefact out of focus (area)": 5,
#     "artefact pen marking (area)": 6,
# }
# print("Constructing darwin")
# annotations = SlideAnnotations.from_darwin_json(d_fn, z_indices=Z_INDICES, sorting="Z_INDEX")
# annotations2 = WsiAnnotations.from_darwin_json(d_fn, z_indices=Z_INDICES, sorting="Z_INDEX")
# print("Constructing converted")
# annotations3 = SlideAnnotations.from_geojson(fn)
# scaling = 0.02


# def scale_bbox(bbox, scaling):
#     coordinates = (bbox[0][0] * scaling, bbox[0][1] * scaling)
#     size = (bbox[1][0] * scaling, bbox[1][1] * scaling)
#     return coordinates, size


# bbox = scale_bbox(annotations.bounding_box, scaling)

# print(bbox)

# # for polygon in annotations._layers.polygons:
# #     polygon.index = index_map[polygon.label]

# annotations.reindex_polygons(index_map)

# annotations3.reindex_polygons(index_map)

# region = annotations.read_region((0, 0), scaling, bbox[1])
# region2 = annotations2.read_region((0, 0), scaling, bbox[1])
# # region3 = annotations3.read_region((0, 0), scaling, bbox[1])

# print(len(region.polygons))

# LUT = annotations3.color_lut
# # print(LUT)

# mask = LUT[region.to_mask()]

# PIL.Image.fromarray(mask).save("mask.png")

# _, mask_origin, _ = convert_annotations(region2, tuple(map(int, bbox[1]))[::-1], index_map)
# PIL.Image.fromarray(LUT[mask_origin]).save("mask2.png")

# # mask3 = LUT[region3.to_mask()]
# # PIL.Image.fromarray(mask3).save("mask3.png")
