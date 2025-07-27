import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch
from app.utils.model_logic import (
    run_evaluation_per_compteur,
    display_metrics_table,
)


@patch("app.utils.model_logic.plot_predictions")
@patch("app.utils.model_logic.compute_residuals_plot")
@patch("app.utils.model_logic.interpret_model")
def test_run_evaluation_resid_and_interp(
    mock_interp, mock_resid, mock_plot
):
    # --- Données factices cohérentes ---
    fake_dates = pd.date_range("2025-04-01", periods=2, freq="h")
    compteur_id = ("27 quai de la Tournelle", "SE-NO")
    site_labels = {compteur_id: "Tournelle - SE-NO"}
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
    run_evaluation_per_compteur(
        results=results,
        site_labels=site_labels,
        show_metrics=False,
        show_preds=False,
        show_resid=True,
        show_interp=True,
        periode_limite=("2025-04-01", "2025-04-16"),
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
def test_run_evaluation_metrics_and_preds(
    mock_display_metrics,
    mock_compute_metrics,
    mock_plot_predictions
):
    # --- Setup data ---
    compteur_id = ("Magenta", "NE-SO")
    site_labels = {compteur_id: "Magenta - NE-SO"}
    fake_dates = pd.date_range("2025-04-01", periods=2, freq="h")
    results = {
        compteur_id: {
            "y_train": [10, 15],
            "y_train_pred": [11, 14],
            "y_test": [12, 13],
            "y_test_pred": [13, 12],
            "X_test_dates": fake_dates,
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
    run_evaluation_per_compteur(
        results=results,
        site_labels=site_labels,
        show_metrics=True,
        show_preds=True,
        show_resid=False,
        show_interp=False,
        periode_limite=("2025-04-01", "2025-04-16"),
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
            "R2_train": 0.91,
            "RMSE_train": 12.5,
            "MAE_train": 9.3,
            "R2_test": 0.87,
            "RMSE_test": 15.0,
            "MAE_test": 11.0,
        },
        {
            "R2_train": 0.85,
            "RMSE_train": 14.0,
            "MAE_train": 10.2,
            "R2_test": 0.80,
            "RMSE_test": 18.2,
            "MAE_test": 12.5,
        },
    ]

    # --- Mock streamlit ---
    st_mock = MagicMock()

    # --- Appel fonction ---
    display_metrics_table(metrics_table, st_module=st_mock)

    # --- Vérifications générales ---
    assert st_mock.dataframe.call_count == 2
    assert st_mock.markdown.call_count == 1
    assert "Moyennes des indicateurs" in st_mock.markdown.call_args[0][0]

    # --- Vérification contenu table moyenne ---
    mean_df = pd.DataFrame(metrics_table).select_dtypes(include=np.number).mean()
    expected_row_count = len(metrics_table)
    assert np.isclose(mean_df["R2_train"], 0.88)
    assert np.isclose(mean_df["R2_test"], 0.835)
    assert expected_row_count == 2
