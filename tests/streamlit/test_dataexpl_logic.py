import pandas as pd
from io import StringIO
from typing import cast
from unittest.mock import patch, Mock
from app.utils.dataexpl_logic import (
    cached_load_dataset_exploration,
    cached_get_missing_periods,
    manage_dataset_exploration,
)


def test_cached_load_dataset_exploration_uploaded_file(monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    cached_load_dataset_exploration.clear()  # type: ignore
    csv_data = "index,col1\n0,foo\n1,bar"
    uploaded_file = StringIO(csv_data)

    df = cached_load_dataset_exploration(uploaded_file)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["col1"]
    assert df.shape == (2, 1)


@patch("app.utils.dataexpl_logic.load_dataset_from_config")
def test_cached_load_dataset_exploration_from_config(mock_loader, monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    cached_load_dataset_exploration.clear()  # type: ignore
    mock_df = pd.DataFrame({"x": [1]})
    mock_loader.return_value = mock_df

    df = cached_load_dataset_exploration(None)
    df = cast(pd.DataFrame, df)

    mock_loader.assert_called_once()
    assert df.equals(mock_df)


@patch("app.utils.dataexpl_logic.get_missing_periods")
def test_cached_get_missing_periods(mock_loader, monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    cached_load_dataset_exploration.clear()  # type: ignore
    mock_df = pd.DataFrame({"x": [1]})
    mock_loader.return_value = mock_df

    df = cached_get_missing_periods(mock_df)
    df = cast(pd.DataFrame, df)

    mock_loader.assert_called_once()
    assert df.equals(mock_df)


def test_manage_dataset_reload_triggers_clear_and_rerun(monkeypatch):
    monkeypatch.setenv("IS_TESTING", "1")
    cached_load_dataset_exploration.clear()  # type: ignore
    mock_st = Mock()
    mock_st.button.return_value = True  # simulate click
    mock_st.spinner.return_value.__enter__ = lambda s: None
    mock_st.spinner.return_value.__exit__ = lambda s, exc, val, tb: None
    mock_st.rerun = Mock()

    with patch("app.utils.dataexpl_logic."
               "cached_load_dataset_exploration.clear") as mock_clear:
        manage_dataset_exploration(mock_st)

    mock_clear.assert_called_once()
    mock_st.rerun.assert_called_once()


def test_manage_dataset_triggers_error_and_empty_df(monkeypatch):
    monkeypatch.setenv("IS_TESTING", "1")
    cached_load_dataset_exploration.clear()  # type: ignore
    mock_st = Mock()
    mock_st.button.return_value = False  # simulate button not clicked
    mock_st.spinner.return_value.__enter__ = lambda s: None
    mock_st.spinner.return_value.__exit__ = lambda s, exc, val, tb: None
    mock_st.stop = Mock()

    with patch("app.utils.dataexpl_logic."
               "cached_load_dataset_exploration",
               return_value=None):
        df_result = manage_dataset_exploration(mock_st)

    mock_st.error.assert_called_once()
    mock_st.stop.assert_called_once()
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.empty
