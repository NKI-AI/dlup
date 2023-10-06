# Copyright (c) dlup contributors
import json
import warnings
from typing import Any

import numpy as np

from dlup.utils.imports import PYTORCH_AVAILABLE

if PYTORCH_AVAILABLE:
    import torch


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if PYTORCH_AVAILABLE and isinstance(obj, torch.Tensor):
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
