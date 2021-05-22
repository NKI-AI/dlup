# coding=utf-8
# Copyright (c) dlup contributors
import argparse
import logging
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from dlup.cli.slidescore import LabelDownloadSlideScoreParser, parse_api_token
from dlup.cli.utils import build_cli_logger
from dlup.utils.io import write_json
from dlup.utils.slidescore import build_client

logger = logging.getLogger(__name__)


parser_kwargs = {
    "parents": [
        LabelDownloadSlideScoreParser(epilog="Script can be used in combination with `dlup slidescore download-wsis`")
    ],
    "add_help": False,
    "description": "Download labels from SlideScore.",
}

# TODO: This are the names of the shapes.
ANNOSHAPE_TYPES = ["polygon", "rect", "ellipse", "brush", "heatmap"]


# TODO: This is how to actually retrieve the questions. Think about a proper way to do this.
def retrieve_questions(
    slidescore_url: str, api_token: str, study_id: int, disable_certificate_check: bool = False
) -> dict:
    """
    Retrieve the questions for a given study from SlideScore.

    This call requires specific permissions, as it obtains the configuration of a given study.

    Parameters
    ----------
    slidescore_url: str
        Url of the slidescore server e.g.: https://rhpc.nki.nl/slidescore/ (without Api/).
    api_token: str
        SlideScore API token.
    study_id: int
        Study id as used by SlideScore.
    disable_certificate_check : bool

    Returns
    -------

    """
    client = build_client(slidescore_url, api_token, disable_certificate_check)

    # Get the configuration for this study. Requires specific permissions.
    config = client.get_config(study_id)
    scores = config["scores"]
    return scores


def download_labels(
    slidescore_url: str,
    api_token: str,
    study_id: int,
    save_dir: Path,
    email: Optional[str] = None,
    question: Optional[str] = None,
    disable_certificate_check: bool = False,
) -> None:
    # TODO: Add format to docstring
    """
    Downloads all available annotations for a study on SlideScore from
    one specific author and saves them in a JSON file per image.

    Parameters
    __________
    slidescore_url: str
        Url of the slidescore server e.g.: https://rhpc.nki.nl/slidescore/ (without Api/).
    api_token: str
        SlideScore API token.
    study_id: int
        Study id as used by SlideScore.
    save_dir: Path
        Directory to save the labels to.
    email: str, optional
        The author email/name as registered on SlideScore to download those specific annotations.
    question : str
        The question to obtain the labels for. If not given all labels will be obtained.
    disable_certificate_check : bool
        Disable HTTPS certificate check.

    Returns
    -------
    None
    """
    client = build_client(slidescore_url, api_token, disable_certificate_check)

    if not save_dir.is_dir():
        save_dir.mkdir()

    extra_kwargs = {}
    if email is not None:
        extra_kwargs["email"] = email
    if question is not None:
        extra_kwargs["question"] = question

    # TODO: Images should become a class / NamedTuple
    images = client.get_images(study_id)

    for image in tqdm(images):
        image_id = image["id"]
        annotations = client.get_results(study_id, imageid=image_id, **extra_kwargs)

        annotation_data = {
            "image_id": image_id,
            "study_id": image["studyID"],  # TODO: image must become a class / NamedTuple
            "image_name": image["name"],
            "annotations": [],
        }

        for annotation in annotations:
            data = annotation.points
            if not data:
                continue

            annotation_data["annotations"].append(
                {"user": annotation.user, "question": annotation.question, "data": data}
            )

        # Now save this to JSON.
        write_json(save_dir / f"{image_id}.json", annotation_data, indent=2)


def main(args: argparse.Namespace) -> None:
    """Main function that downloads labels from SlideScore.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments passed from the CLI. Run with `-h` to see the required parameters

    Returns
    -------
    None
    """
    build_cli_logger("download_labels", log_to_file=not args.no_log, verbosity_level=args.verbose)
    api_token = parse_api_token(args.token_path)
    download_labels(
        args.slidescore_url,
        api_token,
        args.study_id,
        args.output_dir,
        question=args.question,
        email=args.user,
        disable_certificate_check=args.disable_certificate_check,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parser_kwargs)  # type: ignore
    cli_args = parser.parse_args()
    main(cli_args)
