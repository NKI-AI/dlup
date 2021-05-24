# coding=utf-8
# Copyright (c) DLUP Contributors
import warnings
from dlup.transforms.helpers import DlupTransform, convert_numpy_to_tensor, _TORCHVISION_AVAILABLE

if _TORCHVISION_AVAILABLE:
    import torchvision.transforms as transforms
else:
    warnings.warn("torchvision is not installed. Wrapped torchvision transforms are not available.")


FORBIDDEN_KEYS = ["mask", "points", "bbox", "contour"]


class TvRandomGrayscale(DlupTransform):
    """Dlup wrapped version of RandomGrayscale. Note that the examples might require modifications.

    **Original docs**:
    Randomly convert image to grayscale with a probability of p (default 0.1).
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        p (float): probability that image should be converted to grayscale.

    Returns:
        PIL Image or Tensor: Grayscale version of the input image with probability p and unchanged
        with probability (1-p).
        - If input image is 1 channel: grayscale version is 1 channel
        - If input image is 3 channel: grayscale version is 3 channel with r == g == b


    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomGrayscale cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomGrayscale(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomErasing(DlupTransform):
    """Dlup wrapped version of RandomErasing. Note that the examples might require modifications.

    **Original docs**:
     Randomly selects a rectangle region in an torch Tensor image and erases its pixels.
    This transform does not support PIL Image.
    'Random Erasing Data Augmentation' by Zhong et al. See https://arxiv.org/abs/1708.04896

    Args:
         p: probability that the random erasing operation will be performed.
         scale: range of proportion of erased area against input image.
         ratio: range of aspect ratio of erased area.
         value: erasing value. Default is 0. If a single int, it is used to
            erase all pixels. If a tuple of length 3, it is used to erase
            R, G, B channels respectively.
            If a str of 'random', erasing each pixel with random values.
         inplace: boolean to make this transform inplace. Default set to False.

    Returns:
        Erased Image.

    Example:
        >>> transform = transforms.Compose([
        >>>   transforms.RandomHorizontalFlip(),
        >>>   transforms.ToTensor(),
        >>>   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        >>>   transforms.RandomErasing(),
        >>> ])

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomErasing cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomErasing(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvLinearTransformation(DlupTransform):
    """Dlup wrapped version of LinearTransformation. Note that the examples might require modifications.

    **Original docs**:
    Transform a tensor image with a square transformation matrix and a mean_vector computed
    offline.
    This transform does not support PIL Image.
    Given transformation_matrix and mean_vector, will flatten the torch.*Tensor and
    subtract mean_vector from it which is then followed by computing the dot
    product with the transformation matrix and then reshaping the tensor to its
    original shape.

    Applications:
        whitening transformation: Suppose X is a column vector zero-centered data.
        Then compute the data covariance matrix [D x D] with torch.mm(X.t(), X),
        perform SVD on this matrix and pass it as transformation_matrix.

    Args:
        transformation_matrix (Tensor): tensor [D x D], D = C x H x W
        mean_vector (Tensor): tensor [D], D = C x H x W

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvLinearTransformation cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.LinearTransformation(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvColorJitter(DlupTransform):
    """Dlup wrapped version of ColorJitter. Note that the examples might require modifications.

    **Original docs**:
    Randomly change the brightness, contrast, saturation and hue of an image.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, mode "1", "L", "I", "F" and modes with transparency (alpha channel) are not supported.

    Args:
        brightness (float or tuple of float (min, max)): How much to jitter brightness.
            brightness_factor is chosen uniformly from [max(0, 1 - brightness), 1 + brightness]
            or the given [min, max]. Should be non negative numbers.
        contrast (float or tuple of float (min, max)): How much to jitter contrast.
            contrast_factor is chosen uniformly from [max(0, 1 - contrast), 1 + contrast]
            or the given [min, max]. Should be non negative numbers.
        saturation (float or tuple of float (min, max)): How much to jitter saturation.
            saturation_factor is chosen uniformly from [max(0, 1 - saturation), 1 + saturation]
            or the given [min, max]. Should be non negative numbers.
        hue (float or tuple of float (min, max)): How much to jitter hue.
            hue_factor is chosen uniformly from [-hue, hue] or the given [min, max].
            Should have 0<= hue <= 0.5 or -0.5 <= min <= max <= 0.5.

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvColorJitter cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.ColorJitter(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvGrayscale(DlupTransform):
    """Dlup wrapped version of Grayscale. Note that the examples might require modifications.

    **Original docs**:
    Convert image to grayscale.
    If the image is torch Tensor, it is expected
    to have [..., 3, H, W] shape, where ... means an arbitrary number of leading dimensions

    Args:
        num_output_channels (int): (1 or 3) number of channels desired for output image

    Returns:
        PIL Image: Grayscale version of the input.
         - If ``num_output_channels == 1`` : returned image is single channel
         - If ``num_output_channels == 3`` : returned image is 3 channel with r == g == b


    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvGrayscale cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.Grayscale(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvNormalize(DlupTransform):
    """Dlup wrapped version of Normalize. Note that the examples might require modifications.

    **Original docs**:
    Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.


    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvNormalize cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.Normalize(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvCenterCrop(DlupTransform):
    """Dlup wrapped version of CenterCrop. Note that the examples might require modifications.

    **Original docs**:
    Crops the given image at the center.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.
    If image size is smaller than output size along any edge, image is padded with 0 and then center cropped.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvCenterCrop cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        for key in ["image", "mask"]:
            data = sample[key]
            self.__check_dimensionality(data, key)
            sample[key] = transforms.CenterCrop(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvPad(DlupTransform):
    """Dlup wrapped version of Pad. Note that the examples might require modifications.

    **Original docs**:
    Pad the given image on all sides with the given "pad" value.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means at most 2 leading dimensions for mode reflect and symmetric,
    at most 3 leading dimensions for mode edge,
    and an arbitrary number of leading dimensions for mode constant

    Args:
        padding (int or sequence): Padding on each border. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a sequence of length 1: ``[padding, ]``.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric.
            Default is constant.

            - constant: pads with a constant value, this value is specified with fill

            - edge: pads with the last value at the edge of the image,
                    if input a 5D torch Tensor, the last 3 dimensions will be padded instead of the last 2

            - reflect: pads with reflection of image without repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

            - symmetric: pads with reflection of image repeating the last value on the edge

                For example, padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvPad cannot be applied to " "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        for key in ["image", "mask"]:
            data = sample[key]
            self.__check_dimensionality(data, key)
            sample[key] = transforms.Pad(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvFiveCrop(DlupTransform):
    """Dlup wrapped version of FiveCrop. Note that the examples might require modifications.

    **Original docs**:
    Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
         size (sequence or int): Desired output size of the crop. If size is an ``int``
            instead of sequence like (h, w), a square crop of size (size, size) is made.
            If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Example:
         >>> transform = Compose([
         >>>    FiveCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvFiveCrop cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        for key in ["image", "mask"]:
            data = sample[key]
            self.__check_dimensionality(data, key)
            sample[key] = transforms.FiveCrop(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvTenCrop(DlupTransform):
    """Dlup wrapped version of TenCrop. Note that the examples might require modifications.

    **Original docs**:
    Crop the given image into four corners and the central crop plus the flipped version of
    these (horizontal flipping is used by default).
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    .. Note::
         This transform returns a tuple of images and there may be a mismatch in the number of
         inputs and targets your Dataset returns. See below for an example of how to deal with
         this.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        vertical_flip (bool): Use vertical flipping instead of horizontal

    Example:
         >>> transform = Compose([
         >>>    TenCrop(size), # this is a list of PIL Images
         >>>    Lambda(lambda crops: torch.stack([ToTensor()(crop) for crop in crops])) # returns a 4D tensor
         >>> ])
         >>> #In your test loop you can do the following:
         >>> input, target = batch # input is a 5d tensor, target is 2d
         >>> bs, ncrops, c, h, w = input.size()
         >>> result = model(input.view(-1, c, h, w)) # fuse batch size and ncrops
         >>> result_avg = result.view(bs, ncrops, -1).mean(1) # avg over crops

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvTenCrop cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        for key in ["image", "mask"]:
            data = sample[key]
            self.__check_dimensionality(data, key)
            sample[key] = transforms.TenCrop(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomCrop(DlupTransform):
    """Dlup wrapped version of RandomCrop. Note that the examples might require modifications.

    **Original docs**:
    Crop the given image at a random location.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions,
    but if non-constant padding is used, the input is expected to have at most 2 leading dimensions

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is None. If a single int is provided this
            is used to pad all borders. If sequence of length 2 is provided this is the padding
            on left/right and top/bottom respectively. If a sequence of length 4 is provided
            this is the padding for the left, top, right and bottom borders respectively.
            In torchscript mode padding as single int is not supported, use a sequence of length 1: ``[padding, ]``.
        pad_if_needed (boolean): It will pad the image if smaller than the
            desired size to avoid raising an exception. Since cropping is done
            after padding, the padding seems to be done at a random offset.
        fill (number or str or tuple): Pixel fill value for constant fill. Default is 0. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the padding_mode is constant.
            Only number is supported for torch Tensor.
            Only int or str or tuple value is supported for PIL Image.
        padding_mode (str): Type of padding. Should be: constant, edge, reflect or symmetric. Default is constant.

             - constant: pads with a constant value, this value is specified with fill

             - edge: pads with the last value on the edge of the image

             - reflect: pads with reflection of image (without repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in reflect mode
                will result in [3, 2, 1, 2, 3, 4, 3, 2]

             - symmetric: pads with reflection of image (repeating the last value on the edge)

                padding [1, 2, 3, 4] with 2 elements on both sides in symmetric mode
                will result in [2, 1, 1, 2, 3, 4, 4, 3]


    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        if any(_ in sample.keys() for _ in FORBIDDEN_KEYS):
            raise ValueError("Cannot apply torchvision RandomCrop due to random changes in the image input geometry.")
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomCrop cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomCrop(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomHorizontalFlip(DlupTransform):
    """Dlup wrapped version of RandomHorizontalFlip. Note that the examples might require modifications.

    **Original docs**:
    Horizontally flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        if any(_ in sample.keys() for _ in FORBIDDEN_KEYS):
            raise ValueError(
                "Cannot apply torchvision RandomHorizontalFlip due to random changes in the image input geometry."
            )
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomHorizontalFlip cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomHorizontalFlip(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomVerticalFlip(DlupTransform):
    """Dlup wrapped version of RandomVerticalFlip. Note that the examples might require modifications.

    **Original docs**:
    Vertically flip the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading
    dimensions

    Args:
        p (float): probability of the image being flipped. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        if any(_ in sample.keys() for _ in FORBIDDEN_KEYS):
            raise ValueError(
                "Cannot apply torchvision RandomVerticalFlip due to random changes in the image input geometry."
            )
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomVerticalFlip cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomVerticalFlip(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomPerspective(DlupTransform):
    """Dlup wrapped version of RandomPerspective. Note that the examples might require modifications.

    **Original docs**:
    Performs a random perspective transformation of the given image with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        distortion_scale (float): argument to control the degree of distortion and ranges from 0 to 1.
            Default is 0.5.
        p (float): probability of the image being transformed. Default is 0.5.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.BILINEAR``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
            If input is PIL Image, the options is only available for ``Pillow>=5.0.0``.

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        if any(_ in sample.keys() for _ in FORBIDDEN_KEYS):
            raise ValueError(
                "Cannot apply torchvision RandomPerspective due to random changes in the image input geometry."
            )
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomPerspective cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomPerspective(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomRotation(DlupTransform):
    """Dlup wrapped version of RandomRotation. Note that the examples might require modifications.

    **Original docs**:
    Rotate the image by angle.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (sequence, optional): Optional center of rotation, (x, y). Origin is the upper left corner.
            Default is the center of the image.
        fill (sequence or number): Pixel fill value for the area outside the rotated
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
            If input is PIL Image, the options is only available for ``Pillow>=5.2.0``.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters


    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        if any(_ in sample.keys() for _ in FORBIDDEN_KEYS):
            raise ValueError(
                "Cannot apply torchvision RandomRotation due to random changes in the image input geometry."
            )
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomRotation cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomRotation(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomAffine(DlupTransform):
    """Dlup wrapped version of RandomAffine. Note that the examples might require modifications.

    **Original docs**:
    Random affine transformation of the image keeping center invariant.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        degrees (sequence or number): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to deactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or number, optional): Range of degrees to select from.
            If shear is a number, a shear parallel to the x axis in the range (-shear, +shear)
            will be applied. Else if shear is a sequence of 2 values a shear parallel to the x axis in the
            range (shear[0], shear[1]) will be applied. Else if shear is a sequence of 4 values,
            a x-axis shear in (shear[0], shear[1]) and y-axis shear in (shear[2], shear[3]) will be applied.
            Will not apply shear by default.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
            For backward compatibility integer values (e.g. ``PIL.Image.NEAREST``) are still acceptable.
        fill (sequence or number): Pixel fill value for the area outside the transformed
            image. Default is ``0``. If given a number, the value is used for all bands respectively.
            If input is PIL Image, the options is only available for ``Pillow>=5.0.0``.
        fillcolor (sequence or number, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``fill`` parameter instead.
        resample (int, optional): deprecated argument and will be removed since v0.10.0.
            Please use the ``interpolation`` parameter instead.

    .. _filters: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#filters


    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        if any(_ in sample.keys() for _ in FORBIDDEN_KEYS):
            raise ValueError(
                "Cannot apply torchvision RandomAffine due to random changes in the image input geometry."
            )
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomAffine cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomAffine(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomInvert(DlupTransform):
    """Dlup wrapped version of RandomInvert. Note that the examples might require modifications.

    **Original docs**:
    Inverts the colors of the given image randomly with a given probability.
    If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being color inverted. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomInvert cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomInvert(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomPosterize(DlupTransform):
    """Dlup wrapped version of RandomPosterize. Note that the examples might require modifications.

    **Original docs**:
    Posterize the image randomly with a given probability by reducing the
    number of bits for each color channel. If the image is torch Tensor, it should be of type torch.uint8,
    and it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        bits (int): number of bits to keep for each channel (0-8)
        p (float): probability of the image being color inverted. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomPosterize cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomPosterize(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomSolarize(DlupTransform):
    """Dlup wrapped version of RandomSolarize. Note that the examples might require modifications.

    **Original docs**:
    Solarize the image randomly with a given probability by inverting all pixel
    values above a threshold. If img is a Tensor, it is expected to be in [..., 1 or 3, H, W] format,
    where ... means it can have an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        threshold (float): all pixels equal or above this value are inverted.
        p (float): probability of the image being color inverted. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomSolarize cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomSolarize(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomAdjustSharpness(DlupTransform):
    """Dlup wrapped version of RandomAdjustSharpness. Note that the examples might require modifications.

    **Original docs**:
    Adjust the sharpness of the image randomly with a given probability. If the image is torch Tensor,
    it is expected to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.

    Args:
        sharpness_factor (float):  How much to adjust the sharpness. Can be
            any non negative number. 0 gives a blurred image, 1 gives the
            original image while 2 increases the sharpness by a factor of 2.
        p (float): probability of the image being color inverted. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomAdjustSharpness cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomAdjustSharpness(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomAutocontrast(DlupTransform):
    """Dlup wrapped version of RandomAutocontrast. Note that the examples might require modifications.

    **Original docs**:
    Autocontrast the pixels of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".

    Args:
        p (float): probability of the image being autocontrasted. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomAutocontrast cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomAutocontrast(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")


class TvRandomEqualize(DlupTransform):
    """Dlup wrapped version of RandomEqualize. Note that the examples might require modifications.

    **Original docs**:
    Equalize the histogram of the given image randomly with a given probability.
    If the image is torch Tensor, it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "P", "L" or "RGB".

    Args:
        p (float): probability of the image being equalized. Default value is 0.5

    """

    def __init__(self, *args, **kwargs):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError("You want to use transforms from `torchvision` which is not installed yet.")

        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def __call__(self, sample):
        sample = convert_numpy_to_tensor(sample)
        # Points and bbox might need a custom function to implement, or a 'poor mans' version converting to a mask.
        if any(_ in sample for _ in ["points", "bbox", "contour"]):
            raise NotImplementedError(
                "TvRandomEqualize cannot be applied to "
                "to any `points`, `bbox` or `contour` without explicit implementation."
            )

        data = sample["image"]
        self.__check_dimensionality(data, "image")
        sample["image"] = transforms.RandomEqualize(*self.args, **self.kwargs)(data)
        return sample

    @staticmethod
    def __check_dimensionality(data, key):
        """
        Check if input is 2D.

        Parameters
        ----------
        data : ArrayLike
        key : str

        Returns
        -------

        """
        if data.ndim != 3:
            raise ValueError(f"torchvision transforms can only be applied to 2D data. Got {data.shape} for `{key}`.")
