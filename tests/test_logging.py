import logging
import tempfile
from pathlib import Path

import pytest

from dlup.logging import build_cli_logger, setup_logging


@pytest.mark.usefixtures("caplog")
class TestLogging:
    def test_setup_logging_valid_log_level(self, caplog):
        with tempfile.NamedTemporaryFile() as tmp_file:
            setup_logging(filename=Path(tmp_file.name), log_level="DEBUG")
            assert len(caplog.records) == 0

    def test_setup_logging_invalid_log_level(self):
        with pytest.raises(ValueError, match="Unexpected log level got INVALID"):
            setup_logging(log_level="INVALID")

    def test_setup_logging_filename_creation(self, tmp_path):
        with tempfile.NamedTemporaryFile() as tmp_file:
            setup_logging(filename=Path(tmp_file.name))
            assert Path(tmp_file.name).exists()

    def test_setup_logging_log_message(self, caplog):
        with tempfile.NamedTemporaryFile() as tmp_file:
            setup_logging(filename=Path(tmp_file.name), log_level="DEBUG")
            logging.debug("This is a debug message.")
            assert caplog.records[0].message == "This is a debug message."

    @pytest.mark.usefixtures("tmp_path")
    class TestCLILogger:
        def test_build_cli_logger_filename_creation(self, tmp_path):
            build_cli_logger("test_logger", True, 1, tmp_path)
            assert any(tmp_path.iterdir())  # checks if any file is created in tmp_path

        def test_build_cli_logger_valid_verbosity(self, caplog, tmp_path):
            build_cli_logger("test_logger", True, 1, tmp_path)
            logging.info("This is an info message.")
            assert caplog.records[-1].message == "This is an info message."

        def test_build_cli_logger_warning_message(self, caplog, tmp_path):
            build_cli_logger("test_logger", True, 1, tmp_path)
            assert (
                caplog.records[0].message
                == "Beta software. In case you run into issues report at https://github.com/NKI-AI/dlup/."
            )
