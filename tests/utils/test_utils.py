# Copyright (c) dlup contributors

import json

import numpy as np
import pytest

from dlup.utils import ArrayEncoder


class TestArrayEncoder:
    def test_encode_numpy_array(self):
        arr = np.array([1, 2, 3, 4, 5])
        result = json.dumps(arr, cls=ArrayEncoder)
        assert result == "[1, 2, 3, 4, 5]"

    def test_large_numpy_array_warning(self):
        large_arr = np.zeros(int(10e4 + 1))
        with pytest.warns(UserWarning, match=r"Trying to JSON serialize a very large array"):
            json.dumps(large_arr, cls=ArrayEncoder)

    def test_encode_numpy_integers(self):
        int32_val = np.int32(42)
        int64_val = np.int64(42)
        result_int32 = json.dumps(int32_val, cls=ArrayEncoder)
        result_int64 = json.dumps(int64_val, cls=ArrayEncoder)
        assert result_int32 == "42"
        assert result_int64 == "42"

    def test_unhandled_data_type(self):
        with pytest.raises(TypeError, match=r"Object of type .* is not JSON serializable"):
            json.dumps({"key": object()}, cls=ArrayEncoder)  # Here `object()` is an unhandled data type
