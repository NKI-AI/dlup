# coding=utf-8
# Copyright (c) DLUP Contributors
import argparse
import csv
import logging
import sys

from dlup.cli.slidescore import LabelUploadSlideScoreParser, parse_api_token
from dlup.utils.slidescore import APIClient, SlideScoreResult

logger = logging.getLogger(__name__)


parser_kwargs = {
    "parents": [LabelUploadSlideScoreParser(epilog="")],
    "add_help": False,
    "description": "Uploads labels to SlideScore.",
}


def main(args: argparse.Namespace) -> None:
    """Main function that uploads labels to SlideScore.

    Parameters
    ----------
    args: argparse.Namespace
        The arguments passed from the CLI. Run with `-h` to see the required parameters

    Returns
    -------
    None
    """
    url = args.slidescore_url
    api_token = parse_api_token(args.token_path)
    study_id = args.study_id
    client = APIClient(url, api_token)
    wsi_results = []

    # Increase csv field size limit for large rows
    try:
        csv.field_size_limit(sys.maxsize)
    except OverflowError:
        csv.field_size_limit(int(sys.maxsize / 10))

    with open(args.results_file, "r") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=args.csv_delimiter, fieldnames=args.csv_fieldnames)
        for row in reader:
            image_id = row["imageID"]
            image_name = row["imageName"]
            user = row["user"] if args.user is None else args.user
            question = row["question"]
            answer = row["answer"].replace("'", '"') + "\n"

            wsi_result = SlideScoreResult(
                {
                    "imageID": image_id,
                    "imageName": image_name,
                    "user": user,
                    "question": question,
                    "answer": answer,
                }
            )
            wsi_results.append(wsi_result)

    client.upload_results(study_id, wsi_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(parser_kwargs)  # type: ignore
    cli_args = parser.parse_args()
    main(cli_args)
