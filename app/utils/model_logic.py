import matplotlib.pyplot as plt
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
                metrics = compute_metrics(res["y_test"], res["y_test_pred"])
                for k, v in metrics.items():
                    st.info(f"**{k}**: {v}")

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
