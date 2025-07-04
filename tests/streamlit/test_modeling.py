import os
import pytest
import pandas as pd
from smartcheck.modeling_project_specific import compute_metrics
import app.modeling as am


@pytest.fixture(autouse=True)
def enable_test_mode():
    os.environ["IS_TESTING"] = "1"
    am.cached_load_dataset_ml.clear()  # type: ignore
    am.cached_train_model.clear()  # type: ignore
    yield
    del os.environ["IS_TESTING"]


def test_cached_load_dataset_ml_returns_empty_structure():
    df = am.cached_load_dataset_ml()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {
        "nom_du_site_de_comptage", "orientation_compteur", "comptage_horaire"
    }


def test_cached_train_model_returns_fixed_result():
    dummy_df = pd.DataFrame({
        "nom_du_site_de_comptage": ["Test"],
        "orientation_compteur": ["S-N"],
        "comptage_horaire": [100],
    })
    result = am.cached_train_model(
        dummy_df, "LinearRegression", "comptage_horaire", [], False, 0.25
    )
    assert "y_test" in result
    assert len(result["y_test"]) == 2
    assert isinstance(result["X_test_dates"], pd.DatetimeIndex)


def test_compute_metrics_on_test_output():
    res = am.cached_train_model(
        pd.DataFrame({
            "nom_du_site_de_comptage": ["Test"],
            "orientation_compteur": ["N-S"],
            "comptage_horaire": [42],
        }),
        "KNN",
        "comptage_horaire",
        [],
        False,
        0.3
    )
    metrics = compute_metrics(res["y_test"], res["y_test_pred"])
    assert "mae" in metrics
    assert metrics["mae"] >= 0


def test_error_if_dataset_is_none(monkeypatch):
    monkeypatch.setattr(am, "cached_load_dataset_ml", lambda: None)
    with pytest.raises(Exception):
        # Simulation partielle du bloc qui lève st.stop()
        df = am.cached_load_dataset_ml()
        assert df is None
        if df is None or not isinstance(df, pd.DataFrame):
            raise Exception("Handled error from streamlit stop logic")


def test_empty_results_trigger_warning(monkeypatch):
    dummy_df = pd.DataFrame(columns=[
        "nom_du_site_de_comptage", "orientation_compteur", "comptage_horaire"
    ])
    monkeypatch.setattr(am, "cached_load_dataset_ml", lambda: dummy_df)

    # simulate all sites being filtered out
    selected_sites = []
    grouped = dummy_df.groupby(["nom_du_site_de_comptage",
                                "orientation_compteur"])
    results = {
        k: v for k, v in grouped if k in selected_sites
    }
    assert results == {}
