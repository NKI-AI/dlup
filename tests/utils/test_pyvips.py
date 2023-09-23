# Copyright (c) dlup contributors
import numpy as np
import pytest

from dlup.utils.pyvips_utils import numpy_to_vips, vips_to_numpy


@pytest.mark.parametrize("shape", [[256, 256, 1], [512, 512, 3]])
@pytest.mark.parametrize("dtype", [float, np.uint8])
def test_vips_conversion(shape, dtype):
    """Test which test vips to numpy functionality by mapping back and forth"""
    array = np.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    vips_image = numpy_to_vips(array)
    np.testing.assert_array_equal(array, vips_to_numpy(vips_image))
