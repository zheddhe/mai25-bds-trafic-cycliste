import os
import pytest
import pandas as pd
from smartcheck.modeling_project_specific import compute_metrics
import app.sections.modeling as am


@pytest.fixture(autouse=True)
def enable_test_mode():
    os.environ["IS_TESTING"] = "1"
    am.cached_load_dataset_ml.clear()  # type: ignore
    am.cached_train_model.clear()  # type: ignore
    yield
    del os.environ["IS_TESTING"]


def test_cached_load_dataset_ml_returns_empty_structure():
    df = am.cached_load_dataset_ml(None)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {
        "nom_du_site_de_comptage", "orientation_compteur", "comptage_horaire"
    }


def test_cached_train_model_returns_fixed_result():
    dummy_df = pd.DataFrame({
        "nom_du_site_de_comptage": ["Totem 73 boulevard de Sébastopol"],
        "orientation_compteur": ["S-N"],
        "comptage_horaire": [100],
    })
    result = am.cached_train_model(
        dummy_df, "LinearRegression", "MinMaxScaler",
        "comptage_horaire", [], False, 0.25, False
    )
    assert "y_test" in result
    assert len(result["y_test"]) == 2
    assert isinstance(result["X_test_dates"], pd.DatetimeIndex)


def test_compute_metrics_on_test_output():
    res = am.cached_train_model(
        pd.DataFrame({
            "nom_du_site_de_comptage": ["Totem 73 boulevard de Sébastopol"],
            "orientation_compteur": ["N-S"],
            "comptage_horaire": [42],
        }),
        "KNN",
        "MinMaxScaler",
        "comptage_horaire",
        [],
        False,
        0.3,
        False
    )
    metrics = compute_metrics(res["y_test"], res["y_test_pred"])
    assert "MAE" in metrics
    assert metrics["MAE"] >= 0
