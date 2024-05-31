# Copyright (c) dlup contributors
import numpy as np
import pytest

from dlup.utils.pyvips_utils import numpy_to_vips


@pytest.mark.parametrize("shape", [[256, 256, 1], [512, 512, 3]])
@pytest.mark.parametrize("dtype", [float, np.uint8])
def test_vips_conversion(shape, dtype):
    """Test which test vips to numpy functionality by mapping back and forth"""
    array = np.arange(0, np.prod(shape)).reshape(shape).astype(dtype)
    vips_image = numpy_to_vips(array)
    image_array = np.asarray(vips_image)
    if image_array.ndim == 2:
        image_array = image_array[..., np.newaxis]

    np.testing.assert_array_equal(array, image_array)
