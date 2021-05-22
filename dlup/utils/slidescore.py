# coding=utf-8
# Copyright (c) DLUP Contributors
import io
import json
import logging
import pathlib
import re
import shutil
import sys
import urllib.parse
from typing import Dict, List, Optional, Tuple, Union

import requests
from PIL import Image
from requests import Response
from tqdm import tqdm

type_to_name = [
    "FreeText",
    "Integer",
    "Real",
    "DropDown",
    "PositiveNegative",
    "Intensities",
    "Percentage",
    "Checkbox",
    "ClickFriendly",
    "PercentageOnly",
    "IntensitiesOnly",
    "PositiveNegativeOnly",
    "ClickFriendlyOnly",
    "ClickFriendlyPollOnly",
    "Explanation",
    "HScore",
    "HScoreOnly",
    "AnnoPoints",
    "AnnoMeasure",
    "AnnoShapes",
]


class SlideScoreResult:
    """Slidescore wrapper class for storing SlideScore server responses."""

    def __init__(self, slide_dict: Dict = None):
        """
        Parameters
        ----------
        slide_dict : dict
            SlideScore server response for annotations/labels.
        """
        self.__slide_dict = slide_dict
        if not slide_dict:
            slide_dict = {
                "imageID": 0,
                "imageName": "",
                "user": None,
                "tmaRow": None,
                "tmaCol": None,
                "tmaSampleID": None,
                "question": None,
                "answer": None,
            }

        self.image_id = int(slide_dict["imageID"])
        self.image_name = slide_dict["imageName"]
        self.user = slide_dict["user"]
        self.tma_row = int(slide_dict["tmaRow"]) if "tmaRow" in slide_dict else None
        self.tma_col = int(slide_dict["tmaCol"]) if "tmaCol" in slide_dict else None
        self.tma_sample_id = slide_dict["tmaSampleID"] if "tmaSampleID" in slide_dict else ""
        self.question = slide_dict["question"]
        self.answer = slide_dict["answer"]

        self.points = None
        if self.answer is not None and self.answer[:2] == "[{":
            annos = json.loads(self.answer)
            if len(annos) > 0:
                if hasattr(annos[0], "type"):
                    self.annotations = annos
                else:
                    self.points = annos

    def to_row(self) -> str:
        if self.__slide_dict is None:
            return ""
        ret = str(self.image_id) + "\t" + self.image_name + "\t" + self.user + "\t"
        if self.tma_row is not None:
            ret = ret + str(self.tma_row) + "\t" + str(self.tma_col) + "\t" + self.tma_sample_id + "\t"
        ret = ret + self.question + "\t" + self.answer
        return ret

    def __repr__(self):
        return (
            f"SlideScoreResult(image_id={self.image_id}, "
            f"image_name={self.image_name}, "
            f"user={self.user}, "
            f"tma_row={self.tma_row}, "
            f"tma_col={self.tma_col}, "
            f"tma_sample_id={self.tma_sample_id}, "
            f"question={self.question}, "
            f"answer=length {len(self.answer)})"
        )


class APIClient:
    """SlideScore API client."""

    def __init__(self, server: str, api_token: str, disable_cert_checking: bool = False) -> None:
        """
        Base client class for interfacing with slidescore servers.
        Needs and slidescore_url (example: "https://rhpc.nki.nl/slidescore/"), and a api token. Note the ending "/".

        Parameters
        ----------
        server : str
            Path to SlideScore server (without "Api/").
        api_token : str
            API token for this API request.
        disable_cert_checking : bool
            Disable checking of SSL certification (not recommended).
        """
        self.logger = logging.getLogger(type(self).__name__)

        self.server = server
        self.end_point = urllib.parse.urljoin(server, "Api/")
        self.api_token = api_token
        self.verify_certificate = not disable_cert_checking
        self.base_url: Union[str, None] = None
        self.cookie: Union[str, None] = None

        # TODO: Implement using from requests.auth import HTTPBasicAuth, with wrapper to requests.

    def perform_request(
        self,
        request: str,
        data: Optional[Dict],
        method: str = "POST",
        stream: bool = False,
    ) -> Response:
        """
        Base functionality for making requests to slidescore servers. Request should\
        be in the format of the slidescore API: https://www.slidescore.com/docs/api.html

        Parameters
        ----------
        request : str
        data : dict
        method : str
            HTTP request method (POST or GET).
        stream : bool

        Returns
        -------
        Response
        """
        if method not in ["POST", "GET"]:
            raise SlideScoreErrorException(f"Expected method to be either `POST` or `GET`. Got {method}.")

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_token}",
        }
        url = urllib.parse.urljoin(self.end_point, request)

        if method == "POST":
            response = requests.post(url, verify=self.verify_certificate, headers=headers, data=data)
        else:
            response = requests.get(
                url,
                verify=self.verify_certificate,
                headers=headers,
                data=data,
                stream=stream,
            )
        if response.status_code != 200:
            response.raise_for_status()

        return response

    def get_images(self, study_id: int) -> Dict:
        """
        Get slide data (no slides) for all slides in the study.

        Parameters
        ----------
        study_id : int

        Returns
        -------
        dict
            Dictionary containing the images in the study.
        """
        # TODO: Convert to NamedTuple
        response = self.perform_request("Images", {"studyid": study_id})
        rjson = response.json()
        self.logger.info(f"found {len(rjson)} slides with SlideScore API for study ID {study_id}.")

        return rjson

    def download_slide(
        self, study_id: int, image: dict, save_dir: pathlib.Path, skip_if_exists: bool = True, verbose: bool = True
    ) -> pathlib.Path:
        """
        Downloads a WSI from the SlideScore server, needs study_id and image.
        NOTE: this request has a different signature with different rights as the other
        methods and thus might require an api token with more rights than the other ones.

        Parameters
        ----------
        study_id : int
        image : dict
        save_dir : pathlib.Path
        skip_if_exists : bool
        verbose : bool

        Returns
        -------
        pathlib.Path
            Filename the output has been written to.

        """
        # TODO: Allow to disable verbosity.
        image_id = image["id"]
        filesize = image["fileSize"]
        response = self.perform_request(
            "DownloadSlide",
            {"studyid": study_id, "imageid": image_id},
            method="GET",
            stream=True,
        )

        raw = response.headers["Content-Disposition"]
        filename = self._get_filename(raw)
        self.logger.info(f"Writing to {save_dir / filename} (reporting file size of {filesize})...")
        write_to = save_dir / filename

        if skip_if_exists and write_to.is_file():
            self.logger.info(f"File {save_dir / filename} exists. Skipping.")
            return write_to

        temp_write_to = write_to.with_suffix(write_to.suffix + ".partial")
        # TODO: If the partial file exists, can we continue the stream?
        with tqdm.wrapattr(
            open(temp_write_to, "wb"),
            "write",
            miniters=1,
            desc=filename,
            total=None,
        ) as f:
            for chunk in response.iter_content(chunk_size=4096):
                f.write(chunk)
        shutil.move(str(temp_write_to), str(write_to))
        return write_to

    def get_results(self, study_id: int, **kwargs) -> List[SlideScoreResult]:
        """
        Basic functionality to download all annotations made for a particular study.
        Returns a SlideScoreResult class wrapper containing the information.

        Parameters
        ----------
        study_id : int
            ID of SlideScore study.
        **kwargs: dict
            Dictionary with optional API flags. Check https://www.slidescore.com/docs/api.html#get-results for
            more details.

        Returns
        -------
        List[SlideScoreResult]
            List of SlideScore results.
        """
        optional_keys = ["question", "email", "imageid", "caseid"]
        if any(_ not in optional_keys for _ in kwargs):
            raise RuntimeError(f"Expected optional keys to be any of {', '.join(optional_keys)}. Got {kwargs.keys()}.")

        response = self.perform_request("Scores", {"studyid": study_id, **kwargs})
        rjson = response.json()
        return [SlideScoreResult(r) for r in rjson]

    def get_config(self, study_id: int) -> dict:
        """
        Get the configuration of a particular study. Returns a dictionary.

        Parameters
        ----------
        study_id : int
            ID of SlideScore study.

        Returns
        -------
        dict
        """
        response = self.perform_request("GetConfig", {"studyid": study_id})
        rjson = response.json()

        if not rjson["success"]:
            raise SlideScoreErrorException(f"Configuration for study id {study_id} not returned succesfully")

        return rjson["config"]

    def upload_results(self, study_id: int, results: List[SlideScoreResult]) -> bool:
        """
        Basic functionality to upload all annotations made for a particular study.
        Returns true if successful.

        results should be a list of strings, where each elemement is a line of text of the following format:
        imageID - tab - imageNumber - tab - author - tab - question - tab - annotatations
        annotations should be of the following format: a list of dicts containing annotation answers,
        converted to str format

        Parameters
        ----------
        study_id : int
        results : List[str]

        Returns
        -------
        bool
        """
        results_as_row = [r.to_row() for r in results]
        sres = "\n" + "\n".join(results_as_row)
        response = self.perform_request("UploadResults", {"studyid": study_id, "results": sres})
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException(rjson["log"])
        return True

    def upload_asap(
        self,
        image_id: int,
        user: str,
        questions_map: Dict,
        annotation_name: str,
        asap_annotation: Dict,
    ) -> bool:
        """Upload annotations for study using ASAP functionality."""
        response = self.perform_request(
            "UploadASAPAnnotations",
            {
                "imageid": image_id,
                "questionsMap": "\n".join(key + ";" + val for key, val in questions_map.items()),
                "user": user,
                "annotationName": annotation_name,
                "asapAnnotation": asap_annotation,
            },
        )
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException(rjson["log"])
        return True

    def get_image_metadata(self, image_id: int) -> dict:
        """
        Returns slide metadata for that image id.

        Parameters
        ----------
        image_id : int
            SlideScore Image ID.

        Returns
        -------
        dict
            Image metadata as stored in SlideScore.
        """
        response = self.perform_request("GetImageMetadata", {"imageId": image_id}, "GET")
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException(rjson["log"])
        return rjson["metadata"]

    def export_asap(self, image_id: int, user: str, question: str) -> str:
        """Downloads ASAP annotations."""
        response = self.perform_request(
            "ExportASAPAnnotations",
            {"imageid": image_id, "user": user, "question": question},
        )
        rjson = response.json()
        if not rjson["success"]:
            raise SlideScoreErrorException(rjson["log"])
        rawresp = response.text
        if rawresp[0] != "<":
            raise RuntimeError("Incomplete XML ASAP output.")
        return rawresp

    def get_image_server_url(self, image_id: int) -> Tuple[str, str]:
        """
        Returns the image server slidescore url for given image.

        Parameters
        ----------
        image_id : int
            SlideScore Image ID.

        Returns
        -------
        tuple
            Pair consisting of url, cookie.
        """
        if self.base_url is None:
            raise RuntimeError

        response = self.perform_request(f"GetTileServer?imageId={str(image_id)}", None, method="GET")
        rjson: Dict = dict(response.json())
        url_parts = "/".join(["i", str(image_id), rjson["urlPart"], "_files"])
        return (
            urllib.parse.urljoin(self.base_url, url_parts),
            rjson["cookiePart"],
        )

    def set_cookie(self, image_id: int) -> None:
        """Set cookie; needed for getting deep zoomed tiles.

        Parameters
        ----------
        image_id : int
            SlideScore Image ID.
        """
        (self.base_url, self.cookie) = self.get_image_server_url(image_id)

    def get_tile(self, level: int, x: int, y: int) -> Image:
        """
        Gets tile from WSI for given magnification level.
        A WSI at any given magnification level is converted into an x by y tile matrix. This method downloads the tile
        at col (x) and row (y) only as jpeg. Maximum magnification level can be calculated as follows:
        max_level = int(np.ceil(math.log(max_dim, 2))), where max_dim is is the maximum of either height or width
        of the slide. This can be requested by calling get_image_metadata.

        Parameters
        ----------
        level : int
        x : x
        y : x

        Returns
        -------
        PIL.Image
            Requested tile.
        """
        if self.base_url is None:
            raise RuntimeError

        response = requests.get(
            self.base_url + f"/{str(level)}/{str(x)}_{str(y)}.jpeg",
            stream=True,
            cookies=dict(t=self.cookie),
        )
        if response.status_code == 200:
            return Image.open(io.BytesIO(response.content))
        raise SlideScoreErrorException(f"Expected response code 200. Got {response.status_code}.")

    @staticmethod
    def _get_filename(s: str) -> str:
        """
        Method to extract the filename from the HTTP header.

        Parameters
        ----------
        s : str

        Returns
        -------
        str
            Filename extracted from HTTP header.
        """
        filename = re.findall(r"filename\*?=([^;]+)", s, flags=re.IGNORECASE)
        return filename[0].strip().strip('"')


class SlideScoreErrorException(Exception):
    """Class to hold a SlideScore exception."""

    pass


def build_client(slidescore_url: str, api_token: str, disable_certificate_check: bool = False) -> APIClient:
    """
    Build a SlideScore API Client.

    Parameters
    ----------
    slidescore_url: str
        Url of the slidescore server e.g.: https://rhpc.nki.nl/slidescore/ (without Api/).
    api_token: str
        SlideScore API token.
    disable_certificate_check : bool
        Disable HTTPS certificate check.

    Returns
    -------
    APIClient
        A SlideScore API client.
    """
    try:
        client = APIClient(slidescore_url, api_token, disable_cert_checking=disable_certificate_check)
    except requests.exceptions.SSLError as e:
        sys.exit(
            f"SSLError, possibly because the SSL certificate cannot be read. "
            f"If you know what you are doing you can try --disable-certificate-check. "
            f"Full error: {e}"
        )  # TODO parse it better.
    # TODO: This could also be a file or whatever, needs a bit more of an elaborate check.

    return client
