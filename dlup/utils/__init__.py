# coding=utf-8
# Copyright (c) dlup contributors
import json
import warnings
from typing import Any

import numpy as np

from dlup.utils.imports import _PYTORCH_AVAILABLE

if _PYTORCH_AVAILABLE:
    import torch  # pylint: disable=import-error


class ArrayEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if _PYTORCH_AVAILABLE and isinstance(o, torch.Tensor):
            o = o.numpy()

        if isinstance(o, np.ndarray):
            if o.size > 10e4:
                warnings.warn(
                    "Trying to JSON serialize a very large array of size {obj.size}. "
                    "Consider doing this differently"
                )
            return o.tolist()

        if isinstance(o, (np.int32, np.int64)):
            return int(o)

        return json.JSONEncoder.default(self, o)
