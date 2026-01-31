"""This script converts Slidescore annotations to DLUP XML format.
In addition it plots the annotations using matplotlib.
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt

from dlup import SlideAnnotations
from dlup.geometry import Box, Point, Polygon
from pathlib import Path
from typing import Any, Generator


def draw_slidescore_annotations(annotations: SlideAnnotations) -> None:
    fig, ax = plt.subplots(figsize=(10, 10))

    def iter_layers() -> Generator[Polygon | Box | Point, Any, None]:
        # First returns all the polygons then all points
        for polygon in annotations.polygons:
            yield polygon

        for box in annotations.boxes:
            yield box

        for point in annotations.points:
            yield point

    for geom in iter_layers():
        if isinstance(geom, Point):
            ax.plot(geom.x, geom.y, "ro", markersize=5, label=geom.label)  # Red dot for points

        elif isinstance(geom, Polygon):
            exterior = geom.get_exterior()
            interiors = geom.get_interiors()

            # Draw the exterior polygon
            poly_patch = patches.Polygon(
                exterior, closed=True, edgecolor="blue", facecolor="none", lw=2, label=geom.label
            )
            ax.add_patch(poly_patch)

            # Draw holes if any
            for hole in interiors:
                hole_patch = patches.Polygon(
                    hole, closed=True, edgecolor="red", facecolor="none", lw=1, linestyle="dashed"
                )
                ax.add_patch(hole_patch)

        elif isinstance(geom, Box):
            min_x, min_y = geom.coordinates
            width, height = geom.size
            box_patch = patches.Rectangle(
                (min_x, min_y), width, height, edgecolor="green", facecolor="none", lw=2, label="Box"
            )
            ax.add_patch(box_patch)

    ax.set_aspect("equal")
    plt.xlabel("X Coordinates")
    plt.ylabel("Y Coordinates")
    plt.gca().invert_yaxis()
    plt.title("Annotations Visualization")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # ROI_NAMES = ["Polygon Annotations"]
    ROI_NAMES = None

    # Load original Slidescore test file
    original_annotation_file = Path(__file__).parent / "files" / "slidescore_annotation_test.txt"
    annotations = SlideAnnotations.from_file_path(
        original_annotation_file, reader="slidescore_tsv", roi_names=ROI_NAMES
    )
    draw_slidescore_annotations(annotations=annotations)

    # Convert to DLUP XML
    exported_annotation_file_dlup_xml = Path(__file__).parent / "files" / "slidescore_annotation_test_exported_dlup.xml"
    with open(exported_annotation_file_dlup_xml, "w", encoding="utf-8") as f:
        f.write(annotations.as_dlup_xml())
    print("Dlup xml:", exported_annotation_file_dlup_xml)
    exported_annotations_dlup = SlideAnnotations.from_file_path(exported_annotation_file_dlup_xml, reader="dlup_xml")

    # Export as Slidescore tsv again
    exported_annotation_file_slidescore_tsv = (
        Path(__file__).parent / "files" / "slidescore_annotation_test_exported.txt"
    )
    with open(exported_annotation_file_slidescore_tsv, "w", encoding="utf-8") as f:
        f.write(
            exported_annotations_dlup.as_slidescore_tsv(
                image_id=1234, image_name="Test Image", user_email="j.doe@example.com"
            )
        )
    print("Slidescore tsv:", exported_annotation_file_slidescore_tsv)
    exported_annotations_slidescore = SlideAnnotations.from_file_path(
        exported_annotation_file_slidescore_tsv, reader="slidescore_tsv", roi_names=ROI_NAMES, box_as_polygon=True
    )
    draw_slidescore_annotations(annotations=exported_annotations_slidescore)

    # Assertion fails because Ellipses get converted to normal Polygons.
    # Set and get fields do not get exported in DLUP XML
    ellipse_attrs = ["_ellipse_approximation", "_ellipse_center", "_ellipse_size"]
    for polygons in (annotations.polygons, exported_annotations_slidescore.polygons):
        for poly in polygons:
            for attr in ellipse_attrs:
                poly.set_field(attr, None)
    assert annotations == exported_annotations_slidescore
