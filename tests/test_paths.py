import pytest
import importlib.resources
from smartcheck.paths import load_config
from pathlib import Path

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