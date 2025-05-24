import os
import pytest
import importlib.resources
from pathlib import Path
from smartcheck.paths import (
    PROJECT_ROOT,
    load_config, 
    get_full_path    
)

# === Test Class for get_full_path ===
class TestGetFullPath:

    # === Tests ===
    @pytest.mark.parametrize(
        "relative_path",
        [
            "data/file.csv",
            "config/settings.yaml",
            "logs/output.log"
        ]
    )
    def test_get_full_path_resolves_correctly(self, relative_path):
        result = get_full_path(relative_path)
        expected = os.path.join(PROJECT_ROOT, relative_path)
        assert result == expected
        assert result.startswith(PROJECT_ROOT)
        assert relative_path in result

    def test_get_full_path_with_empty_string(self):
        result = get_full_path("")
        assert os.path.normpath(result) == os.path.normpath(PROJECT_ROOT)

# === Test Class for load_config ===
class TestLoadConfig:

    # === Tests ===
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

    def test_yaml_not_a_dict(self, tmp_path, monkeypatch):
        bad_yaml = tmp_path / "config.yaml"
        bad_yaml.write_text("- item1\n- item2", encoding="utf-8")

        class DummyFiles:
            def joinpath(self, filename):
                return bad_yaml

        # create a temporary test mock of the importlib.resources.files to create an error case scenario
        monkeypatch.setattr(importlib.resources, "files", lambda *args, **kwargs: DummyFiles())

        with pytest.raises(ValueError, match="YAML config content must be a dictionary"):
            load_config()

    def test_yaml_invalid_syntax(self, tmp_path, monkeypatch):
        invalid_yaml = tmp_path / "config.yaml"
        invalid_yaml.write_text("key: value: other", encoding="utf-8")

        class DummyFiles:
            def joinpath(self, filename):
                return invalid_yaml

        # create a temporary test mock of the importlib.resources.files to create an error case scenario
        monkeypatch.setattr(importlib.resources, "files", lambda *args, **kwargs: DummyFiles())

        with pytest.raises(ValueError, match="Error parsing YAML"):
            load_config()

    def test_yaml_file_not_found(self, monkeypatch):
        class DummyFiles:
            def joinpath(self, filename):
                return Path("nonexistent.yaml")

        # create a temporary test mock of the importlib.resources.files to create an error case scenario
        monkeypatch.setattr(importlib.resources, "files", lambda *args, **kwargs: DummyFiles())

        with pytest.raises(FileNotFoundError, match="Configuration file 'config.yaml' not found."):
            load_config()