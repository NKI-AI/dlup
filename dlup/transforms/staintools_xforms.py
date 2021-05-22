# coding=utf-8
# Copyright (c) dlup contributors

import numpy as np
import PIL
import staintools
from PIL import Image
from staintools import StainNormalizer

from dlup.transforms import DlupTransform


class CustomMacenkoNormalizer(DlupTransform):
    """Macenko method for H&E normalization. Takes a target tile, and transforms the color space of other tiles
    to the target tile H&E colour distribution

    This uses the staintools implementation

    """

    def __init__(self, path_to_target_im=""):
        self.normalizer = StainNormalizer(method="macenko")
        target_im = Image.open(path_to_target_im)
        target_im = np.array(target_im)
        self.normalizer.fit(target_im)

    def transform(self, im: PIL.Image) -> PIL.Image:
        """Transforms a given image to the initialized target image

        Args:
            im (PIL.Image): Takes a PIL image, as it's developed to be used within a pytorch transforms composition

        Returns:
            PIL.Image: Returns a transforemd PIL Image, so that it can be used further on in the transforms pipeline
        """
        filename = im.filename["tile"]
        im = np.array(im)
        transformed_im = self.normalizer.transform(im)
        return Image.fromarray(transformed_im)
