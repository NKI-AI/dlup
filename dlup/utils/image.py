import math

from dlup import UnsupportedSlideError


def check_if_mpp_is_valid(mpp_x: float, mpp_y: float, *, rel_tol: float = 0.015) -> None:
    """
    Checks if the mpp is (nearly) isotropic and defined. The maximum allowed rel_tol

    Parameters
    ----------
    mpp_x : float
    mpp_y : float
    rel_tol : float
        Relative tolerance between mpp_x and mpp_y. 1.5% seems to work well for most slides.

    Returns
    -------
    None
    """
    if mpp_x == 0 or mpp_y == 0:
        raise UnsupportedSlideError("Unable to parse mpp.")

    if not mpp_x or not mpp_y or not math.isclose(mpp_x, mpp_y, rel_tol=rel_tol):
        raise UnsupportedSlideError(f"cannot deal with slides having anisotropic mpps. Got {mpp_x} and {mpp_y}.")
