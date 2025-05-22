import os
import pytest
from smartcheck.paths import (
    load_config
)

# === Tests for load_config ===
class TestLoadConfig:
    @pytest.mark.parametrize(
        "valid_key, expected_substring",
        [
            ("velib_dispo_data", "https://drive.google.com"),
            ("velib_comptage_data", "https://drive.google.com"),
        ]
    )
    def test_valid_key_contains_expected_substring(self, valid_key, expected_substring):
        config = load_config()
        assert isinstance(config, dict)
        assert expected_substring in config["data"]["input"][valid_key]

    @pytest.mark.parametrize(
        "invalid_key",
        [
            "dummy1",
            "1234567",
        ]
    )
    def test_missing_key_raises_keyerror(self, invalid_key):
        config = load_config()
        assert isinstance(config, dict)
        with pytest.raises(KeyError):
            _ = config["data"]["input"][invalid_key]