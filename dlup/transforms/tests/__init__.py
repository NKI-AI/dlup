# coding=utf-8
# Copyright (c) dlup contributors
import numpy as np
import torch

from dlup.transforms import TvGrayscale
from dlup.transforms._helpers import _wrap_tv

# def test_tv_resize():
#     resize_transform = TvResize(size=(15, 15))
#     sample = {"image": np.random.random((3, 15, 20)), "mask": np.ones((3, 15, 20))}
#     output = resize_transform(sample)
#
#     assert list(output["image"].shape) == [3, 15, 15]
#     assert list(output["mask"].shape) == [3, 15, 15]
