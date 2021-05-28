# coding=utf-8
# Copyright (c) dlup contributors
import logging


class WsiTileDataset:
    def __init__(self, wsi_filename):
        self.logger = logging.getLogger(type(self).__name__)
        self.wsi_filename = wsi_filename
