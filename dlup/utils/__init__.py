# coding=utf-8
# Copyright (c) dlup contributors
import json
import warnings

import numpy as np
import torch

from dlup.utils.imports import _PYTORCH_AVAILABLE

if _PYTORCH_AVAILABLE:
    import torch


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if _PYTORCH_AVAILABLE and isinstance(obj, torch.Tensor):
            obj = obj.numpy()

        if isinstance(obj, np.ndarray):
            if obj.size > 10e4:
                warnings.warn(
                    "Trying to JSON serialize a very large array of size {obj.size}. "
                    "Consider doing this differently"
                )
            return obj.tolist()

        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)

        return json.JSONEncoder.default(self, obj)
