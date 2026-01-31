# Copyright 2024 AI for Oncology Research Group. All Rights Reserved.
# Copyright 2024 Jonas Teuwen. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from datetime import datetime
from typing import Optional

import dlup
from dlup.utils.annotations_utils import rgb_to_hex
from dlup.utils.geometry_xml import create_xml_geometries, create_xml_rois
from dlup.utils.schemas.generated import DlupAnnotations as XMLDlupAnnotations
from dlup.utils.schemas.generated import Metadata as XMLMetadata
from dlup.utils.schemas.generated import Tag as XMLTag
from dlup.utils.schemas.generated import Tags as XMLTags
from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig
from xsdata.models.datatype import XmlDate


def dlup_xml_exporter(
    cls: "dlup.annotations.SlideAnnotations",
    authors: Optional[list[str]] = None,
    indent: Optional[int] = 2,
) -> str:
    """
    Output the annotations as DLUP XML.
    This format supports the complete serialization of a SlideAnnotations object.

    Parameters
    ----------
    authors : list[str], optional
        Authors of the annotations.
    indent : int, optional
        Indent for pretty printing.

    Returns
    -------
    str
        The output as a DLUP XML string.
    """
    image_id = cls.metadata.get("image_id", None)
    if image_id is not None and not isinstance(image_id, str):
        raise ValueError("Image ID must be a string or None.")

    description = cls.metadata.get("description", None)
    if description is not None and not isinstance(description, str):
        raise ValueError("Description must be a string or None.")

    version = cls.metadata.get("version", None)
    if version is not None and not isinstance(version, str):
        raise ValueError("Version must be a string or None.")

    metadata = XMLMetadata(
        image_id=image_id,
        description=description,
        version=version,
        authors=XMLMetadata.Authors(authors) if authors is not None else None,
        date_created=XmlDate.from_string(datetime.now().strftime("%Y-%m-%d")),
        software=f"dlup {dlup.__version__}",
    )
    xml_tags: list[XMLTag] = []
    if cls.tags:
        for tag in cls.tags:
            if tag.attributes:
                attrs = [
                    XMLTag.Attribute(value=_.label, color=rgb_to_hex(*_.color) if _.color else None)
                    for _ in tag.attributes
                ]
            xml_tag = XMLTag(
                attribute=attrs if tag.attributes else [],
                label=tag.label,
                color=rgb_to_hex(*tag.color) if tag.color else None,
            )
            xml_tags.append(xml_tag)

    tags = XMLTags(tag=xml_tags) if xml_tags else None

    geometries = create_xml_geometries(cls)
    rois = create_xml_rois(cls)

    extra_annotation_params: dict[str, XMLTags] = {}
    if tags:
        extra_annotation_params["tags"] = tags

    dlup_annotations = XMLDlupAnnotations(
        metadata=metadata, geometries=geometries, regions_of_interest=rois, **extra_annotation_params
    )
    config = SerializerConfig(pretty_print=True)
    serializer = XmlSerializer(config=config)
    return str(serializer.render(dlup_annotations))
