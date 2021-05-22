# coding=utf-8
# Copyright (c) dlup contributors
import os

from dlup.utils.slidescore import APIClient


def test_slidescore_api():
    slidescore_api_token = os.environ.get("SLIDESCORE_API_KEY", None)
    if not slidescore_api_token:
        raise ValueError("SLIDESCORE_API_KEY environment variable not set.")

    url = "https://rhpc.nki.nl/slidescore"
    client = APIClient(url, slidescore_api_token)


if __name__ == "__main__":
    test_slidescore_api()
