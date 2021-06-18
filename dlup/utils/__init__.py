# coding=utf-8
# Copyright (c) dlup contributors
import json
import warnings
import subprocess
import sys
import numpy as np
import torch


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            obj = obj.numpy()

        if isinstance(obj, np.ndarray):
            if obj.size > 10e4:
                warnings.warn(
                    "Trying to JSON serialize a very large array of size {obj.size}. "
                    "Reconsider doing this differently"
                )
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
