from streamlit.testing.v1 import AppTest
from unittest.mock import patch
import pandas as pd


@patch("smartcheck.dataframe_common.load_dataset_from_config")
def test_data_exploration_with_success(mock_loader):
    df_fake = pd.DataFrame({"a": range(5), "b": range(5)})
    mock_loader.return_value = df_fake

    at = AppTest.from_file("app/data_exploration.py")
    at.run()

    assert any("Dataset chargé avec succès" in s.value for s in at.success)
    assert len(at.dataframe) > 0


@patch("smartcheck.dataframe_common.load_dataset_from_config")
def test_data_exploration_with_failure(mock_loader):
    mock_loader.return_value = None

    at = AppTest.from_file("app/data_exploration.py")
    at.run()

    assert any("Impossible de charger les données" in e.value for e in at.error)
