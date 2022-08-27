# coding=utf-8
# Copyright (c) dlup contributors

"""Test the transforms.
"""
from pprint import pformat

import numpy as np
from torchvision.transforms import Compose

from dlup.annotations import WsiAnnotations
from dlup.data.dataset import TiledROIsSlideImageDataset
from dlup.data.transforms import ContainsPolygonToLabel, ConvertAnnotationsToMask, MajorityClassToLabel
from dlup.experimental_backends import ImageBackend
from dlup.tiling import TilingMode
from dlup.viz.plotting import plot_2d

annotations = WsiAnnotations.from_asap_xml("100B.xml")
rois = annotations["roi"].bounding_boxes

# Available labels are sorted alphabetically.
# Note that this way might not be robust across different samples as some labels might be missing.
translate_dict = {k: idx + 1 for idx, k in enumerate(annotations.available_labels)}
mask_colors = ("green", "red", "blue", "purple", "orange", "black", "brown")

print("Available labels", annotations.available_labels)
print(f"Translation dict {pformat(translate_dict)}")


transforms = Compose(
    [
        ConvertAnnotationsToMask(roi_name="roi", index_map=translate_dict),
        MajorityClassToLabel(roi_name="roi", index_map=translate_dict),
        ContainsPolygonToLabel(roi_name="roi", label="rest", threshold=0.05),
    ]
)


dataset = TiledROIsSlideImageDataset.from_standard_tiling(
    "100B.tif",
    mpp=1.0,
    tile_size=(1024, 1024),
    tile_overlap=(0, 0),
    mask=None,
    rois=rois,
    tile_mode=TilingMode.overflow,
    annotations=annotations,
    transform=transforms,
    backend=ImageBackend.PYVIPS,
)

samples = [_ for idx, _ in enumerate(dataset) if idx < 10]
images = [
    np.asarray(
        plot_2d(sample["image"], mask=sample["annotation_data"]["mask"], mask_colors=mask_colors, mask_alpha=30)
    )
    for sample in samples
]
