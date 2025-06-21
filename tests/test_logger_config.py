import logging
import pytest
from unittest.mock import patch
from smartcheck.logger_config import setup_logger


# === Test class for setup_logger ===
class TestSetupLogger:

    # === Data fixtures ===
    @pytest.fixture(autouse=True)
    def reset_logging(self):
        """
        Reset logging configuration before each test.
        """
        logging.shutdown()
        import importlib
        importlib.reload(logging)
        yield
        logging.shutdown()
        importlib.reload(logging)

    # === Tests ===
    @patch("smartcheck.logger_config.logging.config.dictConfig")
    def test_logger_called_with_correct_config(self, mock_dict_config):
        setup_logger(logging.DEBUG)
        assert mock_dict_config.called
        config_arg = mock_dict_config.call_args[0][0]
        assert isinstance(config_arg, dict)
        assert config_arg["version"] == 1
        assert config_arg["root"]["level"] == logging.DEBUG

    def test_logger_outputs_info_message(self, caplog):
        caplog.set_level(logging.INFO)
        setup_logger(logging.INFO)
        root_logger = logging.getLogger()
        root_logger.addHandler(caplog.handler)
        logging.info("🚲 Test info message")
        assert "🚲 Test info message" in caplog.text

    def test_logger_respects_debug_level(self, caplog):
        caplog.set_level(logging.DEBUG)
        setup_logger(logging.DEBUG)
        root_logger = logging.getLogger()
        root_logger.addHandler(caplog.handler)
        logging.debug("🐞 Debug message")
        assert "🐞 Debug message" in caplog.text

    def test_logger_does_not_output_below_level(self, caplog):
        caplog.set_level(logging.WARNING)
        setup_logger(logging.WARNING)
        root_logger = logging.getLogger()
        root_logger.addHandler(caplog.handler)
        logging.info("This should not appear")
        logging.warning("⚠️ This should appear")
        assert "This should not appear" not in caplog.text
        assert "⚠️ This should appear" in caplog.text
