import pandas as pd
import numpy as np
from io import StringIO
from typing import cast
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch, Mock
from app.utils.model_logic import (
    cached_load_dataset_ml,
    cached_train_model,
    manage_dataset_modeling,
    display_report_per_counter,
    display_metrics_table,
    manage_training,
)


def test_cached_load_dataset_ml_uploaded_file(monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    cached_load_dataset_ml.clear()  # type: ignore
    csv_data = "index,col1\n0,foo\n1,bar"
    uploaded_file = StringIO(csv_data)

    df = cached_load_dataset_ml(uploaded_file)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["col1"]
    assert df.shape == (2, 1)


@patch("app.utils.model_logic.load_dataset_from_config")
def test_cached_load_dataset_ml_from_config(mock_loader, monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    cached_load_dataset_ml.clear()  # type: ignore
    mock_df = pd.DataFrame({"x": [1]})
    mock_loader.return_value = mock_df

    df = cached_load_dataset_ml(None)
    df = cast(pd.DataFrame, df)

    mock_loader.assert_called_once()
    assert df.equals(mock_df)


@patch("app.utils.model_logic.train_timeseries_model")
def test_cached_train_model(mock_trainer, monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    mock_result = {"first": 1, "second": 2}
    mock_trainer.return_value = mock_result
    df_mock = pd.DataFrame({
        "feature": [1, 2, 3],
        "comptage_horaire": [100, 120, 140]
    })
    mock_train_inputs = {
        "df_compteur": df_mock,
        "model_type": "linear",
        "scaler_type": "standard",
        "target_col": "comptage_horaire",
        "drop_columns": [],
        "temp_feats": [1, 1, 1],
        "test_ratio": 0.2,
        "forecast": False
    }
    results = cached_train_model(**mock_train_inputs)

    mock_trainer.assert_called_once()
    assert results == mock_result


def test_manage_dataset_reload_triggers_clear_and_rerun(monkeypatch):
    monkeypatch.setenv("IS_TESTING", "1")
    mock_st = Mock()
    mock_st.button.return_value = True  # simulate click
    mock_st.spinner.return_value.__enter__ = lambda s: None
    mock_st.spinner.return_value.__exit__ = lambda s, exc, val, tb: None
    mock_st.rerun = Mock()
    with patch("app.utils.model_logic."
               "cached_load_dataset_ml.clear") as mock_clear:
        manage_dataset_modeling(mock_st)
        mock_clear.assert_called_once()
        mock_st.rerun.assert_called_once()


@patch("app.utils.model_logic.plot_predictions")
@patch("app.utils.model_logic.compute_residuals_plot")
@patch("app.utils.model_logic.interpret_model")
def test_display_report_resid_and_interp(
    mock_interp, mock_resid, mock_plot
):
    # --- Données factices cohérentes ---

    train_config = {
        "show_metrics": False,
        "show_preds": False,
        "show_resid": True,
        "show_interp": True,
        "selected_dates": ("2025-04-01", "2025-04-16"),
    }
    compteur_id = ('102 boulevard de Magenta', 'SE-NO')
    fake_dates = pd.date_range("2025-04-01", periods=2, freq="h")
    fake_result = {
        "y_test": [1, 2],
        "y_test_pred": [1.1, 1.9],
        "X_test_dates": fake_dates,
        "date_et_heure_de_comptage_local": fake_dates
    }
    results = {compteur_id: fake_result}

    # --- Préparation des mocks de figures ---
    fig1 = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    fig4 = plt.figure()

    mock_plot.return_value = fig1
    mock_resid.return_value = (fig2, fig3, 0.123)
    mock_interp.return_value = [fig4]

    # --- Mock streamlit ---
    st_mock = MagicMock()
    st_mock.expander.return_value.__enter__.return_value = True

    # --- Exécution ---
    display_report_per_counter(
        results=results,
        train_config=train_config,
        st_module=st_mock,
    )

    # --- Assertions ---
    mock_resid.assert_called_once()
    mock_interp.assert_called_once()
    assert st_mock.pyplot.call_count == 3  # fig2 (resid), fig3 (resid), fig4 (interp)
    assert st_mock.info.call_args[0][0].startswith("Dérive des résidus")

    # --- Nettoyage matplotlib ---
    for fig in [fig1, fig2, fig3, fig4]:
        plt.close(fig)


@patch("app.utils.model_logic.plot_predictions")
@patch("app.utils.model_logic.compute_metrics")
@patch("app.utils.model_logic.display_metrics_table")
def test_display_report_metrics_and_preds(
    mock_display_metrics,
    mock_compute_metrics,
    mock_plot_predictions
):
    # --- Setup data ---
    train_config = {
        "show_metrics": True,
        "show_preds": True,
        "show_resid": False,
        "show_interp": False,
        "selected_dates": ("2025-04-01", "2025-04-16"),
    }
    compteur_id = ('102 boulevard de Magenta', 'SE-NO')
    fake_dates = pd.date_range("2025-04-01", periods=2, freq="h")
    df_fake_dates = pd.DataFrame({
        "date_et_heure_de_comptage_local": [fake_dates]
    })
    results = {
        compteur_id: {
            "y_train": [10, 15],
            "y_train_pred": [11, 14],
            "y_test": [12, 13],
            "y_test_pred": [13, 12],
            "X_test_dates": df_fake_dates,
        }
    }

    # --- Mocks streamlit + fig ---
    st_mock = MagicMock()
    st_mock.expander.return_value.__enter__.return_value = True

    fig = plt.figure()
    mock_plot_predictions.return_value = fig

    mock_compute_metrics.side_effect = [
        {"R2": 0.9, "RMSE": 1.2, "MAE": 0.7},
        {"R2": 0.8, "RMSE": 1.5, "MAE": 0.9},
    ]

    # --- Run function ---
    display_report_per_counter(
        results=results,
        train_config=train_config,
        st_module=st_mock,
    )

    # --- Assert metrics ---
    assert mock_compute_metrics.call_count == 2
    assert mock_display_metrics.called
    assert st_mock.markdown.call_args_list[0][0][0].startswith("### 📈 Métriques")

    # --- Assert predictions ---
    assert mock_plot_predictions.called
    st_mock.pyplot.assert_called_once_with(fig)

    # --- Cleanup ---
    plt.close(fig)


def test_display_metrics_table_basic():
    # --- Données de test ---
    metrics_table = [
        {
            "R² train": 0.91,
            "RMSE train": 12.5,
            "MAE train": 9.3,
            "R² test": 0.87,
            "RMSE test": 15.0,
            "MAE test": 11.0,
        },
        {
            "R² train": 0.85,
            "RMSE train": 14.0,
            "MAE train": 10.2,
            "R² test": 0.80,
            "RMSE test": 18.2,
            "MAE test": 12.5,
        },
    ]

    # --- Mock streamlit ---
    st_mock = MagicMock()

    # --- Appel fonction ---
    display_metrics_table(metrics_table, st_module=st_mock)

    # --- Vérifications générales ---
    assert st_mock.dataframe.call_count == 2

    # --- Vérification contenu table moyenne ---
    mean_df = pd.DataFrame(metrics_table).select_dtypes(include=np.number).mean()
    expected_row_count = len(metrics_table)
    assert np.isclose(mean_df["R² train"], 0.88)
    assert np.isclose(mean_df["R² test"], 0.835)
    assert expected_row_count == 2


@patch("app.utils.model_logic.apply_percent_range_selection")
@patch("app.utils.model_logic.compute_metrics")
@patch("app.utils.model_logic.cached_train_model")
def test_manage_training_single_site(
    mock_cached_train_model,
    mock_compute_metrics,
    mock_apply_range,
):
    # --- Arrange ---
    compteur_id = ('102 boulevard de Magenta', 'SE-NO')
    # compteur_name = "Magenta_SE-NO"
    fake_df = pd.DataFrame({
        "nom_du_site_de_comptage": [compteur_id[0]] * 4,
        "orientation_compteur": [compteur_id[1]] * 4,
        "comptage_horaire": [100, 150, 130, 160]
    })

    train_config = {
        "selected_sites": [compteur_id],
        "range": (10, 90),
        "model": "linear",
        "scaler": "standard",
        "drop_cols": [],
        "ar_nb": 3,
        "mm_nb": 2,
        "mm_season": 1,
        "split": 0.8,
        "use_forecast": False
    }

    mock_df_selected = fake_df.copy()
    mock_apply_range.return_value = mock_df_selected

    mock_cached_train_model.return_value = {
        "y_train": [100, 150],
        "y_train_pred": [98, 151],
        "y_test": [130, 160],
        "y_test_pred": [128, 159]
    }

    mock_compute_metrics.side_effect = [
        {"R2": 0.95, "RMSE": 2.0, "MAE": 1.5},
        {"R2": 0.90, "RMSE": 2.5, "MAE": 2.0}
    ]

    # --- Act ---
    results, metrics_table = manage_training(train_config, fake_df)

    # --- Assert ---
    assert compteur_id in results
    assert isinstance(metrics_table, list)
    assert len(metrics_table) == 1
    assert metrics_table[0]["R² train"] == 0.95
    assert metrics_table[0]["R² test"] == 0.90
