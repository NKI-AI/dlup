# Copyright (c) dlup contributors
import numpy as np
import numpy.typing as npt
import PIL.Image
import pyvips

NUMPY_DTYPE_TO_VIPS_FORMAT = {
    "uint8": "uchar",
    "int8": "char",
    "uint16": "ushort",
    "int16": "short",
    "uint32": "uint",
    "int32": "int",
    "float32": "float",
    "float64": "double",
    "complex64": "complex",
    "complex128": "dpcomplex",
}
VIPS_FORMAT_TO_NUMPY_DTYPE = {
    "uchar": np.uint8,
    "char": np.int8,
    "ushort": np.uint16,
    "short": np.int16,
    "uint": np.uint32,
    "int": np.int32,
    "float": np.float32,
    "double": np.float64,
    "complex": np.complex64,
    "dpcomplex": np.complex128,
}


def numpy_to_vips(data: npt.NDArray[np.int_]) -> pyvips.Image:
    """
    Convert a numpy array to a pyvips image.

    Parameters
    ----------
    data : np.ndarray

    Returns
    -------
    pyvips.Image
    """

    if data.ndim == 2:
        height, width = data.shape
        bands = 1
    else:
        height, width, bands = data.shape

    vips_image = pyvips.Image.new_from_memory(
        np.ascontiguousarray(data).data,
        width,
        height,
        bands,
        NUMPY_DTYPE_TO_VIPS_FORMAT[str(data.dtype)],
    )
    return vips_image


def vips_to_numpy(vips_image: pyvips.Image) -> npt.NDArray[np.int_]:
    """
    Convert a pyvips image to a numpy array.

    Parameters
    ----------
    vips_image : pyvips.Image

    Returns
    -------
    np.ndarray
    """
    return np.ndarray(
        buffer=vips_image.write_to_memory(),
        dtype=VIPS_FORMAT_TO_NUMPY_DTYPE[vips_image.format],
        shape=[vips_image.height, vips_image.width, vips_image.bands],
    )


def pil_to_vips(pil_image: PIL.Image.Image) -> pyvips.Image:
    """
    Convert a PIL image to a pyvips image.

    Parameters
    ----------
    pil_image : PIL.Image.Image

    Returns
    -------
    pyvips.Image
    """
    np_image = np.array(pil_image)
    vips_image = numpy_to_vips(np_image)

    return vips_image


def vips_to_pil(vips_image: pyvips.Image) -> PIL.Image.Image:
    """
    Convert a pyvips image to a PIL image.

    Parameters
    ----------
    vips_image : pyvips.Image

    Returns
    -------
    PIL.Image.Image
    """
    # Use the existing vips_to_numpy function to convert the pyvips image to a NumPy array
    np_image = vips_to_numpy(vips_image)

    # Convert the NumPy array to a PIL image
    if np_image.shape[2] == 1:  # Grayscale image
        np_image = np_image.reshape((np_image.shape[0], np_image.shape[1]))
    pil_image = PIL.Image.fromarray(np_image)

    return pil_image
