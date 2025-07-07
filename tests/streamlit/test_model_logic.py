import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import MagicMock, patch
from app.utils.model_logic import run_evaluation_per_compteur


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
