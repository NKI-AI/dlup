# coding=utf-8
# Copyright (c) dlup contributors
"""
Convert TIGER (https://tiger.grand-challenge.org) dataset annotations to GeoJSON annotations. In case of TCGA data,
these labels are scaled appropriately (The TCGA part of TIGER is downsampled).
"""
import argparse
import json
from pathlib import Path

from tqdm import tqdm

from dlup import SlideImage, UnsupportedSlideError
from dlup.experimental_annotations import WsiAnnotations
from dlup.experimental_backends import ImageBackends


def convert_asap_to_geojson(asap_xml, tiger_image=None, tcga_mpp=None, remap_labels=None) -> dict:
    scaling = 1.0
    if tcga_mpp is not None:
        with SlideImage.from_file_path(tiger_image, backend=ImageBackends.PYVIPS) as slide:
            scaling = slide.mpp / tcga_mpp

    annotation = WsiAnnotations.from_asap_xml(asap_xml, scaling=scaling, remap_labels=remap_labels)
    return annotation.as_geojson(split_per_label=True)


def get_tiger_pairs(annotations_dir, images_dir):
    tiger_pairs = {}
    all_annotations = annotations_dir.glob("TCGA*.xml")
    for annotation_fn in all_annotations:
        image_fn = images_dir / annotation_fn.with_suffix(".tif").name
        if not image_fn.is_file():
            continue
        image_id = image_fn.name[:-4]
        tiger_pairs[image_id] = (annotation_fn, image_fn)
    return tiger_pairs


def get_tcga_mpps(tcga_dir, tiger_pairs):
    tcga_mpps = {}
    all_svs = tcga_dir.glob("**/*.svs")
    for svs_fn in tqdm(all_svs):
        if not svs_fn.name[:-4] in tiger_pairs:
            continue

        try:
            with SlideImage.from_file_path(svs_fn, backend=ImageBackends.OPENSLIDE) as f:
                tcga_mpps[svs_fn.name[:-4]] = f.mpp
        except UnsupportedSlideError as e:
            tqdm.write(f"{e}. Skipping.")

    return tcga_mpps


if __name__ == "__main__":
    # TIGER xml example: "TIGER/wsirois/wsi-level-annotations/annotations-tissue-bcss-xmls"
    # TIGER images = "TIGER/wsirois/wsi-level-annotations/images"
    # TCGA image example: "gdc_manifest.2021-11-01_diagnostic_breast.txt"

    parser = argparse.ArgumentParser(description="Parse TIGER annotations to GeoJSON")
    parser.add_argument(
        "TIGER_ANNOTATION_DIR", help="Directory pointing to TIGER ASAP XML annotations", type=Path
    )
    parser.add_argument("TIGER_IMAGES_DIR", help="Directory pointing to TIGER ASAP tif's", type=Path)
    parser.add_argument("TCGA_IMAGES_DIR", help="Directory pointing to TCGA images", type=Path)
    parser.add_argument("OUTPUT_DIR", help="Where to write the GeoJSON", type=Path)
    args = parser.parse_args()

    tiger_pairs = get_tiger_pairs(annotations_dir=args.TIGER_ANNOTATION_DIR, images_dir=args.TIGER_IMAGES_DIR)
    print(f"Determining mpp's for corresponding TCGA images...")
    tcga_maps = get_tcga_mpps(args.TCGA_IMAGES_DIR, tiger_pairs)
    print(f"Writing new annotations...")

    remap_labels = {
            "exclude": "ignore",
            "tumor-associated stroma": "stroma",
            "invasive tumor": "tumor",
            "inflamed stroma": "inflamed",
            "healthy glands": "irrelevant",
            "necrosis not in-situ": "irrelevant",
            "in-situ tumor": "irrelevant",
            "rest": "rest",
            "roi": "roi"
        }

    for image_id in tqdm(tcga_maps):
        annotation_fn, image_fn = tiger_pairs[image_id]
        geojsons = convert_asap_to_geojson(
            annotation_fn, tiger_image=image_fn, tcga_mpp=tcga_maps[image_id], remap_labels=remap_labels
        )

        write_dir = args.OUTPUT_DIR / image_id
        write_dir.mkdir(exist_ok=True, parents=True)
        for label, geojson in geojsons:
            with open(write_dir / f"{label}.json", "w") as json_file:
                json.dump(geojson, json_file, indent=2)

    print("Completed.")
