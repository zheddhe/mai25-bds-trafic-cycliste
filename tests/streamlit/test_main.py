import importlib.util
import os
import pytest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

PAGES_DIR = Path(__file__).resolve().parent.parent.parent / "app" / "sections"
APP_FILE = Path(__file__).resolve().parent.parent.parent / "app" / "main.py"

PAGES_TO_TEST = [
    "home.py",
    "data_visualization.py",
    "modeling.py",
    "data_exploration.py",
]


@pytest.mark.parametrize("filename", PAGES_TO_TEST)
@patch("smartcheck.dataframe_common.load_dataset_from_config", return_value=None)
def test_streamlit_page_loads(mock_loader, filename):
    os.environ["IS_TESTING"] = "1"  # Activation du mode test
    path = PAGES_DIR / filename
    spec = importlib.util.spec_from_file_location("page_module", path)
    assert spec is not None, f"spec is None for file: {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"spec.loader is None for file: {path}"
    spec.loader.exec_module(module)
    del os.environ["IS_TESTING"]  # Nettoyage


def test_app_home_title_displayed():
    os.environ["IS_TESTING"] = "1"
    at = AppTest.from_file(str(APP_FILE))
    at.run()
    titles = [t.value for t in at.title]
    assert "🚲 Application Trafic Cycliste" in titles
    del os.environ["IS_TESTING"]
