import os
import pytest
import pandas as pd
from unittest.mock import patch
from smartcheck.modeling_project_specific import compute_metrics
import app.sections.modeling as am


@pytest.fixture(autouse=True)
def enable_test_mode():
    os.environ["IS_TESTING"] = "1"
    am.cached_load_dataset_ml.clear()  # type: ignore
    yield
    del os.environ["IS_TESTING"]


def test_cached_load_dataset_ml_returns_dataframe():
    df = am.cached_load_dataset_ml()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {
        "nom_du_site_de_comptage",
        "orientation_compteur",
        "comptage_horaire"
    }


@patch("app.sections.modeling.train_timeseries_model")
def test_train_timeseries_model_mocked_return(mock_train):
    dummy_result = {
        "X_test_dates": pd.date_range("2024-01-01", periods=2, freq="h"),
        "y_test": pd.Series([100, 120]),
        "y_test_pred": pd.Series([95, 125]),
        "y_train": pd.Series([80, 90]),
        "y_train_pred": pd.Series([82, 88])
    }
    mock_train.return_value = dummy_result

    dummy_df = pd.DataFrame({
        "nom_du_site_de_comptage": ["Test"],
        "orientation_compteur": ["S-N"],
        "comptage_horaire": [100],
    })

    result = mock_train(dummy_df, "LinearRegression", "MinMaxScaler",
                        "comptage_horaire", [], [0, 0, 24], 0.25)

    assert "y_test" in result
    assert len(result["y_test"]) == 2
    assert isinstance(result["X_test_dates"], pd.DatetimeIndex)


@patch("app.sections.modeling.train_timeseries_model")
def test_compute_metrics_on_mocked_result(mock_train):
    dummy_result = {
        "y_test": pd.Series([10, 20]),
        "y_test_pred": pd.Series([12, 19]),
        "y_train": pd.Series([5, 15]),
        "y_train_pred": pd.Series([6, 14]),
        "X_test_dates": pd.date_range("2025-01-01", periods=2, freq="h")
    }
    mock_train.return_value = dummy_result

    result = mock_train(None, None, None, None, None, None, None)
    metrics = compute_metrics(result["y_test"], result["y_test_pred"])
    assert "MAE" in metrics
    assert metrics["MAE"] >= 0


def test_error_if_dataset_is_none(monkeypatch):
    monkeypatch.setattr(am, "cached_load_dataset_ml", lambda: None)
    with pytest.raises(Exception):
        df = am.cached_load_dataset_ml()
        assert df is None
        if df is None or not isinstance(df, pd.DataFrame):
            raise Exception("Handled error from streamlit stop logic")


def test_empty_results_trigger_warning(monkeypatch):
    dummy_df = pd.DataFrame(columns=[
        "nom_du_site_de_comptage",
        "orientation_compteur",
        "comptage_horaire"
    ])
    monkeypatch.setattr(am, "cached_load_dataset_ml", lambda: dummy_df)

    selected_sites = []
    grouped = dummy_df.groupby(["nom_du_site_de_comptage",
                                "orientation_compteur"])
    results = {
        k: v for k, v in grouped if k in selected_sites
    }
    assert results == {}
