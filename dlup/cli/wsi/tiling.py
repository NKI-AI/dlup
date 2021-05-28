# coding=utf-8
# Copyright (c) DLUP Contributors
import argparse
import logging
import sys

from dlup.cli.utils import build_cli_logger
from dlup.cli.wsi import WsiPreprocessingParser
from dlup.preprocessors.wsi.preprocessor import WsiBasePreprocessor
from dlup.preprocessors.wsi.slide_manager import SlideManager as WsiSlideManager

logger = logging.getLogger(__name__)


parser_kwargs = {
    "parents": [WsiPreprocessingParser(epilog="")],
    "add_help": False,
    "description": "Dump WSIs to disk in tiles.",
}


def main(args: argparse.Namespace) -> None:
    """Main function that processes WSIs into tiles..

    Parameters
    ----------
    args: argparse.Namespace
        The arguments passed from the CLI. Run with `-h` to see the required parameters

    Returns
    -------
    None
    """
    local_rank = args.local_rank
    build_cli_logger(
        "preprocess_wsi" if not local_rank else f"preprocess_wsi_{local_rank}",
        log_to_file=not args.no_log,
        verbosity_level=args.verbose,
    )

    logger.info("Starting the preprocessing of the whole slide images.")

    slide_manager = WsiSlideManager(**vars(args))
    num_files = len(slide_manager)

    if num_files == 0:
        raise sys.exit(
            f"Slide manager has not found any files with extension {args.extension} "
            f"in the source directory: {args.root_dir}."
        )

    logger.info(f"Initialized SlideManager with {num_files} files.")

    if args.compression != "png":
        raise NotImplementedError(
            "Saving to anything else than png is not supported. The code might support this,"
            "but will require figuring out how the compression and other saving flags are"
            "connected to this."
        )

    try:
        preprocessor = WsiBasePreprocessor(**vars(args))
        preprocessor.run(slide_manager)
    except KeyboardInterrupt:
        logger.warning("Program interrupted by user.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parser_kwargs)  # type: ignore
    cli_args = parser.parse_args()

    # Debug
    import pathlib

    cli_args.input_dir = pathlib.Path("/Users/jonas/Downloads/data/tcga")
    cli_args.output_dir = pathlib.Path("/Users/jonas/Downloads/data/tcga/tiles")

    cli_args.tile_size = 1024
    cli_args.tile_overlap = 0
    cli_args.mpp = 1.0
    cli_args.local_rank = 0
    cli_args.no_log = True
    cli_args.verbose = True
    cli_args.extension = ".svs"
    cli_args.compression = "png"

    main(cli_args)
