# coding=utf-8
# Copyright (c) dlup contributors
import logging
import pathlib
import sys
from typing import Optional


def setup_logging(
    filename: Optional[pathlib.Path] = None,
    use_stdout: bool = True,
    log_level: str = "INFO",
    formatter_str: str = "[%(asctime)s | %(name)s | %(levelname)s] - %(message)s",
) -> None:
    """
    Setup logging for dlup.

    Parameters
    ----------
    use_stdout : bool
        Write output to standard out.
    filename : PathLike
        Filename to write log to.
    log_level : str
        Logging level as in the `python.logging` library.
    formatter_str : str
        Formatting string to be used for logging messages.
    Returns
    -------
    None
    """
    if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "EXCEPTION"]:
        raise ValueError(f"Unexpected log level got {log_level}.")

    logging.captureWarnings(True)
    log_level = getattr(logging, log_level)

    root = logging.getLogger("")
    root.setLevel(log_level)

    if use_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        stdout_formatter = logging.Formatter(formatter_str)
        handler.setFormatter(stdout_formatter)
        root.addHandler(handler)

    if filename:
        filename.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        formatter = logging.Formatter(formatter_str)
        fh.setFormatter(formatter)
        root.addHandler(fh)
