# coding=utf-8
# Copyright (c) DLUP Contributors
import datetime
import logging
import pathlib

from dlup.utils.logging import setup_logging
from dlup.utils.types import PathLike


def build_cli_logger(
    name: str,
    log_to_file: bool,
    verbosity_level: int,
    log_directory: PathLike = ".",
) -> None:
    """
    Setup logging for DLUP.

    Parameters
    ----------
    name : str
        Human readable identifier for the current log.
    log_to_file : bool
        Whether to save log as a file on top of having it on stdout.
    verbosity_level: int
        How verbose the log should be. 0 is least verbose, 2 is most verbose.
    log_directory : PathLike
        Directory to save log file in.
    Returns
    -------
    None
    """
    log_filename = pathlib.Path(log_directory) / pathlib.Path(
        f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}.log"
    )
    levels = ["WARNING", "INFO", "DEBUG"]
    log_level = levels[min(len(levels) - 1, verbosity_level)]
    setup_logging(filename=log_filename if log_to_file else None, log_level=log_level)
    logging.warning("Beta software. In case you run into issues report at https://github.com/NKI-AI/DLUP/.")
