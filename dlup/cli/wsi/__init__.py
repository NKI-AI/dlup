# coding=utf-8
# Copyright (c) DLUP Contributors
import argparse
import pathlib

from dlup.cli.parsers import ArgumentParserWithDefaults


class BasePreprocessingParser(ArgumentParserWithDefaults):
    """Defines global default arguments."""

    def __init__(self, epilog=None, **overrides):
        """
        Parameters
        ----------
            **overrides: dict, optional
                Keyword arguments used to override default argument values.
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawTextHelpFormatter, add_help=False)  # noqa
        # TODO: If you support nargs, you can have non-square tile versions.
        self.add_argument(
            "--tile-size",
            type=int,
            required=True,
            help="Size of the requested output (square) tile size. If the tiles are at boundaries, "
            "the size might be below this format.",
        )
        self.add_argument(
            "--tile-overlap",
            type=int,
            required=True,
            help="Number of px overlap for tile creation.",
        )
        self.add_argument(
            "--local_rank",
            type=int,
            default=None,
            help="Local rank of the process. Used for logging",
        )
        self.add_argument(
            "input_dir",
            type=pathlib.Path,
            help="Input directory.",
        )
        self.add_argument(
            "output_dir",
            type=pathlib.Path,
            help="Directory to save output too.",
        )

        self.set_defaults(**overrides)


class WsiPreprocessingParser(BasePreprocessingParser):
    """Argument parser for WSI tiling."""

    def __init__(self, epilog=None, **overrides):
        """
        Parameters
        ----------
            **overrides: dict, optional
                Keyword arguments used to override default argument values.
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawTextHelpFormatter, add_help=False)  # noqa

        self.add_argument(
            "--mpp",
            type=float,
            default=None,
            help="Number of microns per pixel.",
        )
        self.add_argument(
            "--s-idx-start",
            type=int,
            default=0,
            help="Which index of all whole slide images in the given source directory to start with. "
            "Useful in a distributed setting (e.g. Slurm) in conjunction with --s-idx-end.",
        )
        self.add_argument(
            "--s-idx-end",
            type=int,
            default=-1,
            help="Which index of all whole slide images in the given source directory to end with."
            "if -1 or larger than the maximum, will go over all. "
            "Useful in a distributed setting (e.g. Slurm) in conjunction with --s-idx-start.",
        )
        self.add_argument(
            "--extension",
            type=str,
            default=".svs",
            help="All files in the root directory with given extension will be included in the preprocessing.",
        )
        self.add_argument(
            "--background-threshold",
            type=float,
            default=0.05,
            help="Threshold to determine whether a tile is considered background or not, "
            "based on the fraction of non-zero pixels in the computed tissue mask. "
            "This is a value between 0 and 1. Around 0.05 is typically used.",
        )
        # From large image
        self.add_argument(
            "--compression",
            "-c",
            choices=[
                "",
                "png",
                "jpeg",
                "deflate",
                "zip",
                "lzw",
                "zstd",
                "packbits",
                "jbig",
                "lzma",
                "webp",
                "jp2k",
                "none",
            ],
            default="png",  # TODO: this needs some work
            help="Internal compression.  Default will use jpeg if the source "
            "appears to be lossy or lzw if lossless.  lzw is the most compatible "
            "lossless mode.  jpeg is the most compatible lossy mode.  jbig and "
            "lzma may not be available.  jp2k will first write the file with no "
            "compression and then rewrite it with jp2k the specified psnr or "
            "compression ratio.",
        )
        self.add_argument(
            "--quality",
            "-q",
            type=int,
            default=95,
            help="JPEG or webp compression quality. For webp, specify 0 for " "lossless.",
        )
        self.add_argument(
            "--num-workers",
            type=int,
            default=0,
            help="Number of workers to extract tiles.",
        )
        self.set_defaults(**overrides)


class WsiMetadataParser(ArgumentParserWithDefaults):
    """
    Defines global default arguments.
    """

    def __init__(self, epilog=None, **overrides):
        """
        Parameters
        ----------
            **overrides: dict, optional
                Keyword arguments used to override default argument values.
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawTextHelpFormatter, add_help=False)  # noqa
        self.add_argument("--extension", type=str, default="svs")
        self.add_argument(
            "--save-full-metadata", action="store_true", help="If set, will store the complete properties object."
        )
        self.add_argument(
            "input_dir",
            type=pathlib.Path,
            help="Input directory.",
        )
        self.add_argument(
            "output_file",
            type=pathlib.Path,
            help="File to save output JSON to.",
        )
        self.set_defaults(**overrides)
