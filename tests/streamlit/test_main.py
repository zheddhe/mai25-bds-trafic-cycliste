import importlib.util
import importlib
import os
import pytest
from pathlib import Path
from unittest.mock import patch
from streamlit.testing.v1 import AppTest
from app.utils import dataviz_logic, dataexpl_logic

PAGES_DIR = Path("app/sections/")
APP_FILE = Path("app/main.py")

PAGES_TO_TEST = [
    "home.py",
    "data_visualization.py",
    "modeling.py",
    "data_exploration.py",
    "project_presentation.py",
]


@pytest.mark.parametrize("filename", PAGES_TO_TEST)
def test_streamlit_page_loads(filename):
    os.environ["IS_TESTING"] = "1"
    if filename == "data_visualization.py":
        dataviz_logic.cached_load_dataset_visualization.clear()  # type: ignore
    if filename == "data_exploration.py":
        dataexpl_logic.cached_load_dataset_exploration.clear()  # type: ignore
        dataexpl_logic.cached_get_missing_periods.clear()  # type: ignore
    path = PAGES_DIR / filename
    spec = importlib.util.spec_from_file_location("page_module", path)
    assert spec is not None, f"spec is None for file: {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"spec.loader is None for file: {path}"
    spec.loader.exec_module(module)
    del os.environ["IS_TESTING"]


def test_app_home_error_when_file_missing():
    at = AppTest.from_file(str(APP_FILE))
    with patch("app.config.PAGES_DIR", Path("/fake/pages")), \
         patch("pathlib.Path.exists", return_value=False), \
         patch("streamlit.error") as mock_error:
        at.run()
    expected_path = Path("/fake/pages/home.py")
    mock_error.assert_called_once_with(
        f"❌ Page 'home' not found at {expected_path}"
    )


def test_app_home_title_displayed():
    at = AppTest.from_file(str(APP_FILE))
    at.run()
    titles = [t.value for t in at.title]
    assert "🚲 Application Trafic Cycliste" in titles
