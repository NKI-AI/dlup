# Copyright (c) dlup contributors
from datetime import datetime
from typing import Optional

from xsdata.formats.dataclass.serializers import XmlSerializer
from xsdata.formats.dataclass.serializers.config import SerializerConfig
from xsdata.models.datatype import XmlDate

import dlup
from dlup.utils.annotations_utils import rgb_to_hex
from dlup.utils.geometry_xml import create_xml_geometries, create_xml_rois
from dlup.utils.schemas.generated import DlupAnnotations as XMLDlupAnnotations
from dlup.utils.schemas.generated import Metadata as XMLMetadata
from dlup.utils.schemas.generated import Tag as XMLTag
from dlup.utils.schemas.generated import Tags as XMLTags


def dlup_xml_exporter(
    cls: "dlup.annotations.SlideAnnotations",
    image_id: Optional[str] = None,
    description: Optional[str] = None,
    version: Optional[str] = None,
    authors: Optional[list[str]] = None,
    indent: Optional[int] = 2,
) -> str:
    """
    Output the annotations as DLUP XML.
    This format supports the complete serialization of a SlideAnnotations object.

    Parameters
    ----------
    image_id : str, optional
        The image ID corresponding to this annotation.
    description : str, optional
        Description of the annotations.
    version : str, optional
        Version of the annotations.
    authors : list[str], optional
        Authors of the annotations.
    indent : int, optional
        Indent for pretty printing.

    Returns
    -------
    str
        The output as a DLUP XML string.
    """

    metadata = XMLMetadata(
        image_id=image_id if image_id is not None else "",
        description=description if description is not None else "",
        version=version if version is not None else "",
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

    geometries = create_xml_geometries(cls._layers)
    rois = create_xml_rois(cls._layers)

    extra_annotation_params: dict[str, XMLTags] = {}
    if tags:
        extra_annotation_params["tags"] = tags

    dlup_annotations = XMLDlupAnnotations(
        metadata=metadata, geometries=geometries, regions_of_interest=rois, **extra_annotation_params
    )
    config = SerializerConfig(pretty_print=True)
    serializer = XmlSerializer(config=config)
    return serializer.render(dlup_annotations)
