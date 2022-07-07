import numpy as np

from dlup import UnsupportedSlideError


def check_if_mpp_is_isotropic(mpp_x: float, mpp_y: float) -> None:
    """
    Checks if the mpp is (nearly) isotropic and defined.

    Parameters
    ----------
    mpp_x : float
    mpp_y : float

    Returns
    -------
    None
    """
    if not mpp_x or not mpp_y or not np.isclose(mpp_x, mpp_y, rtol=1.0e-2):
        raise UnsupportedSlideError(f"cannot deal with slides having anisotropic mpps. Got {mpp_x} and {mpp_y}.")
