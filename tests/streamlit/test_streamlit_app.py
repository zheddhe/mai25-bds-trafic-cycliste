import importlib.util
import os
import pytest
from unittest.mock import patch
from streamlit.testing.v1 import AppTest

PAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "app")

PAGES_TO_TEST = [
    "home.py",
    "data_visualization.py",
    "modeling.py",
    "data_exploration.py",  # contient un appel à load_dataset_from_config
]

@pytest.mark.parametrize("filename", PAGES_TO_TEST)
@patch("smartcheck.dataframe_common.load_dataset_from_config", return_value=None)
def test_streamlit_page_loads(mock_loader, filename):
    path = os.path.join(PAGES_DIR, filename)
    spec = importlib.util.spec_from_file_location("page_module", path)
    assert spec is not None, f"spec is None for file: {path}"
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None, f"spec.loader is None for file: {path}"
    spec.loader.exec_module(module)


def test_app_home_title_displayed():
    at = AppTest.from_file("streamlit_app.py")
    at.run()
    titles = [t.value for t in at.title]
    assert "🚲 Projet Trafic Cycliste" in titles
