# coding=utf-8
# Copyright (c) DLUP Contributors
import argparse
import logging
import pathlib

from tqdm import tqdm

from dlup.cli.slidescore import WsiSlideScoreParser, parse_api_token
from dlup.cli.utils import build_cli_logger
from dlup.utils.slidescore import build_client

logger = logging.getLogger(__name__)

parser_kwargs = {
    "parents": [
        WsiSlideScoreParser(epilog="Script can be used in combination with `dlup slidescore download-labels`")
    ],
    "add_help": False,
    "description": "Download WSIs from SlideScore.",
}


def append_to_manifest(save_dir: pathlib.Path, image_id: int, filename: pathlib.Path) -> None:
    """
    Create a manifest mapping image id to the filename.

    Parameters
    ----------
    save_dir : pathlib.Path
    image_id : int
    filename : pathlib.Path

    Returns
    -------
    None
    """
    with open(save_dir / "slidescore_mapping.txt", "a") as file:
        file.write(f"{image_id} {filename.name}\n")


def download_wsis(
    slidescore_url: str, api_token: str, study_id: int, save_dir: pathlib.Path, disable_certificate_check: bool = False
) -> None:
    """
    Download all WSIs for a given study from SlideScore

    Parameters
    ----------
    slidescore_url : str
    api_token : str
    study_id : int
    save_dir : pathlib.Path
    disable_certificate_check : bool

    Returns
    -------
    None
    """
    logger.info(f"Will write to: {save_dir}")
    # Set up client and directories
    client = build_client(slidescore_url, api_token, disable_certificate_check)
    if not save_dir.is_dir():
        save_dir.mkdir()

    # Collect image metadata
    images = client.get_images(study_id)

    # Download and save WSIs
    for image in tqdm(images):
        image_id = image["id"]

        logger.info(f"Downloading image for id: {image_id}")
        filename = client.download_slide(study_id, image, save_dir=save_dir, verbose=True)
        logger.info(f"Image with id {image_id} has been saved to {filename}.")
        append_to_manifest(save_dir, image_id, filename)


def main(args: argparse.Namespace):
    """Main function that downloads WSIs from SlideScore.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments passed from the CLI. Run with `-h` to see the required parameters

    Returns
    -------
    None
    """
    build_cli_logger("download_wsis", log_to_file=not args.no_log, verbosity_level=args.verbose)
    api_token = parse_api_token(args.token_path)
    download_wsis(
        args.slidescore_url,
        api_token,
        args.study_id,
        args.output_dir,
        disable_certificate_check=args.disable_certificate_check,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parser_kwargs)  # type: ignore
    cli_args = parser.parse_args()
    main(cli_args)
