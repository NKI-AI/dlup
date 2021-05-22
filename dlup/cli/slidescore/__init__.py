# coding=utf-8
# Copyright (c) DLUP Contributors
"""SlideScore CLI module. Implements downloading of labels and images and uploading of annotations."""
import argparse
import logging
import os
import pathlib
import sys
from typing import Optional

from dlup.cli.parsers import ArgumentParserWithDefaults
from dlup.utils.types import PathLike

logger = logging.getLogger("slidescore_cli")


class BaseSlideScoreParser(ArgumentParserWithDefaults):
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
        self.add_argument(
            "--slidescore-url",
            type=str,
            help="URL for SlideScore",
            default="https://slidescore.nki.nl/",
        )
        self.add_argument(
            "-t",
            "--token-path",
            dest="token_path",
            type=str,
            required=os.environ.get("SLIDESCORE_API_KEY", None) is None,
            help="Path to file with API token. Required if SLIDESCORE_API_KEY environment variable is not set. "
            "Will overwrite the environment variable if set.",
        )
        self.add_argument(
            "-s",
            "--study",
            dest="study_id",
            help="SlideScore Study ID",
            type=int,
            required=True,
        )
        self.add_argument(
            "--disable-certificate-check",
            help="Disable the certificate check.",
            action="store_true",
        )

        self.set_defaults(**overrides)


class WsiSlideScoreParser(BaseSlideScoreParser):
    """Parser for the CLI tool to download WSIS from SlideScore."""

    def __init__(self, epilog=None, **overrides):
        """
        Parameters
        ----------
            epilog: str, optional
            **overrides: dict, optional
                Keyword arguments used to override default argument values.
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawTextHelpFormatter, add_help=False)  # noqa
        self.add_argument(
            "output_dir",
            type=pathlib.Path,
            help="Directory to save output too.",
        )
        self.set_defaults(**overrides)


class LabelDownloadSlideScoreParser(BaseSlideScoreParser):
    """Parser for the CLI tool to download labels from SlideScore."""

    def __init__(self, epilog=None, **overrides):
        """
        Parameters
        ----------
            epilog: str, optional
            **overrides: dict, optional
                Keyword arguments used to override default argument values.
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawTextHelpFormatter, add_help=False)  # noqa
        self.add_argument(
            "-q",
            "--question",
            dest="question",
            help="Question to save annotations for. If not set, will return all questions.",
            type=str,
            required=False,
        )
        self.add_argument(
            "-u",
            "--user",
            dest="user",
            help="Email(-like) reference indicating submitted annotations on slidescore. "
            "If not set, will return questions from all users.",
            type=str,
            required=False,
        )
        self.add_argument(
            "output_dir",
            type=pathlib.Path,
            help="Directory to save output too.",
        )
        self.set_defaults(**overrides)


class LabelUploadSlideScoreParser(BaseSlideScoreParser):
    """Parser for the CLI tool to upload labels to SlideScore."""

    def __init__(self, epilog=None, **overrides):
        """
        Parameters
        ----------
            epilog: str, optional
            **overrides: dict, optional
                Keyword arguments used to override default argument values.
        """
        super().__init__(epilog=epilog, formatter_class=argparse.RawTextHelpFormatter, add_help=False)  # noqa
        self.add_argument(
            "--csv-delimiter",
            type=str,
            help="The delimiter character used in the csv file.",
            default="\t",
        )
        self.add_argument(
            "-u",
            "--user",
            dest="user",
            help="Email(-like) reference indicating submitted annotations on slidescore. "
            "If not set, will use the one included in the results file.",
            type=str,
            required=False,
        )
        self.add_argument(
            "--csv-fieldnames", nargs="*", type=str, default=["imageID", "imageName", "user", "question", "answer"]
        )
        self.add_argument(
            "-r",
            "--results-file",
            type=str,
            required=True,
            help="The results-file should be .csv file, separated with columns "
            "imageID, imageNumber, user, question, answer (or as given by --csv-fieldnames). "
            "User is the email(-like) address to register this "
            "upload as a unique entry, question is the type of cell (e.g.: lymphocytes, tumour cells) that "
            "pertains to the upload and answer contains a list of annotations (e.g.: ellipse, rectangle, polygon) "
            "to be uploaded to slidescore. See the documentation for some examples.",
        )
        self.set_defaults(**overrides)


def parse_api_token(data: Optional[PathLike] = None) -> str:
    """
    Parse the API token from file or from the SLIDESCORE_API_KEY. If file is given, this will overwrite the
    environment variable.

    Parameters
    ----------
    data : str or pathlib.Path

    Returns
    -------
    str
        SlideScore API Token.
    """
    if data is not None and pathlib.Path(data).is_file():
        # load token
        with open(data, "r") as file:
            api_token = file.read().strip()
    else:
        api_token = os.environ.get("SLIDESCORE_API_KEY", "")

    if not api_token:
        logging.error(
            "SlideScore API token not properly set. "
            "Either pass API token file, or set the SLIDESCORE_API_KEY environmental variable."
        )
        sys.exit()
    return api_token
