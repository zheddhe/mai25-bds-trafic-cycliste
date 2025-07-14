import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from smartcheck.modeling_project_specific import (
    compute_metrics,
    plot_predictions,
    compute_residuals_plot,
    interpret_model,
)


def run_evaluation_per_compteur(results, site_labels,
                                show_metrics, show_preds,
                                show_resid, show_interp,
                                periode_limite,
                                st_module=None):
    """
    Affiche les rapports pour chaque compteur selon les flags fournis.
    Cette fonction est isolée pour faciliter les tests unitaires.

    Args:
        results (dict): Dictionnaire {compteur_id: result_data}.
        site_labels (dict): Dictionnaire {compteur_id: label}.
        show_metrics (bool): Afficher les métriques.
        show_preds (bool): Afficher les prédictions.
        show_resid (bool): Afficher les résidus.
        show_interp (bool): Afficher l'interprétation.
        periode_limite (tuple): Période d'affichage (start_date, end_date).
    """

    st = st_module or __import__("streamlit")

    for compteur_id, res in results.items():
        label = site_labels[compteur_id]
        with st.expander(f"📉 Rapport pour {label}"):

            if show_metrics:
                st.markdown("### 📈 Métriques")
                metrics_table = []
                train_metrics = compute_metrics(res["y_train"], res["y_train_pred"])
                test_metrics = compute_metrics(res["y_test"], res["y_test_pred"])
                combined_row = {
                    "R2_train": train_metrics.get("R2", None),
                    "RMSE_train": train_metrics.get("RMSE", None),
                    "MAE_train": train_metrics.get("MAE", None),
                    "R2_test": test_metrics.get("R2", None),
                    "RMSE_test": test_metrics.get("RMSE", None),
                    "MAE_test": test_metrics.get("MAE", None),
                }
                metrics_table.append(combined_row)
                display_metrics_table(metrics_table, st_module=st)

            if show_preds:
                st.markdown("### 🔮 Prédictions")
                fig = plot_predictions(
                    compteur=label,
                    dates=res["X_test_dates"],
                    y_true=res["y_test"],
                    y_pred=res["y_test_pred"],
                    periode_limite=periode_limite,
                )
                st.pyplot(fig)
                plt.close(fig)

            if show_resid:
                st.markdown("### 🧾 Résidus")
                fig1, fig2, slope = compute_residuals_plot(
                    compteur=label,
                    dates=res["X_test_dates"],
                    y_true=res["y_test"],
                    y_pred=res["y_test_pred"],
                    periode_limite=periode_limite,
                )
                st.pyplot(fig1)
                plt.close(fig1)
                st.info(f"Dérive des résidus : pente = {slope:.4f}")
                st.pyplot(fig2)
                plt.close(fig2)

            if show_interp:
                st.markdown("### 🧠 Interprétation")
                interp_figs = interpret_model(label, res)
                if interp_figs:
                    for fig in interp_figs:
                        st.pyplot(fig)
                        plt.close(fig)


def display_metrics_table(metrics_table, st_module=None):
    st = st_module or __import__("streamlit")

    df_metrics = pd.DataFrame(metrics_table)
    styled_df = (
        df_metrics.style
        .format(precision=2)
        .background_gradient(
            subset=["R2_train", "R2_test"],
            cmap="RdYlGn",  # green = good, red = bad
            vmin=0.0,
            vmax=1.0,
        )
        .highlight_max(axis=0,
                       subset=["R2_test", "R2_train"],
                       props="font-weight: bold;")
        .highlight_min(axis=0,
                       subset=["RMSE_test", "RMSE_train"],
                       props="font-weight: bold;")
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)


def get_selected_period(
    default_start,
    default_end,
    min_dt_str,
    max_dt_str,
    label: str,
    st_module=None
):
    st = st_module or __import__("streamlit")

    def to_naive(dt):
        if isinstance(dt, str):
            dt = datetime.fromisoformat(dt)
        return dt.replace(tzinfo=None)

    start_dt = to_naive(default_start)
    end_dt = to_naive(default_end)
    min_dt = to_naive(min_dt_str)
    max_dt = to_naive(max_dt_str)

    return st.slider(
        label,
        min_value=min_dt,
        max_value=max_dt,
        value=(start_dt, end_dt),
        format="YYYY-MM-DD"
    )
