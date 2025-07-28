import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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
                display_metrics_table(metrics_table, st_module=st, show_mean=False)

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
                interp_figs = interpret_model(res)
                if interp_figs:
                    for fig in interp_figs:
                        st.pyplot(fig)
                        plt.close(fig)


def display_metrics_table(metrics_table, st_module=None, show_mean=True):
    st = st_module or __import__("streamlit")

    df_metrics = pd.DataFrame(metrics_table)

    styled_df = (
        df_metrics.style
        .format(precision=3)
        .background_gradient(
            subset=["R2_train", "R2_test"],
            cmap="RdYlGn",  # green = good, red = bad
            vmin=0.0,
            vmax=1.0,
        )
        # .background_gradient(
        #     subset=["RMSE_train", "RMSE_test"],
        #     cmap="Reds", vmin=0.0  # pas besoin de vmax si tu veux relatif
        # )
    )

    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    if show_mean:
        # display averages
        num_cols = df_metrics.select_dtypes(include=np.number).columns
        mean_row = df_metrics[num_cols].mean().to_frame().T
        mean_row.insert(0, "Nombre de compteurs pour les métriques moyennes",
                        df_metrics.shape[0])

        styled_mean = (
            mean_row.style
            .format(precision=4)
            .background_gradient(
                subset=["R2_train", "R2_test"],
                cmap="RdYlGn", vmin=0.0, vmax=1.0,
            )
        )

        st.dataframe(styled_mean, use_container_width=True, hide_index=True)


def display_train_parameters(
    train_config,
    available_columns,
    st_module=None
):
    st = st_module or __import__("streamlit")

    with st.expander("Résumé des paramètres d'entrainement courants",
                     expanded=False):
        col1, col2 = st.columns([1, 1])
        portion = train_config['range'][1] - train_config['range'][0]
        with col1:
            st.markdown(f"""
            - **Modèle utilisé** : `{train_config['model']}`
            - **Mise à l'échelle utilisé** : `{train_config['scaler']}`
            - **Nb d'Auto-régression** : `{train_config['ar_nb']}`
            - **Nb de Moyenne mobile** : `{train_config['mm_nb']}`
            - **Taille de la fenêtre (heures)** : `{train_config['mm_season']}`
            - **Prédiction dynamique des AR/MM** : `{train_config['use_forecast']}`
            - **Portion du dataset d'origine** : `{portion}%`
            (entre `{train_config['range'][0]}%` et
            `{train_config['range'][1]}%`)
            - **Répartition Train/Test** :
            `{int(train_config['split'] * 100)}%` /
            `{int((1 - train_config['split']) * 100)}%`
            - **Plage d'affichage des prédictions** :
            `{train_config['selected_dates'][0].strftime("%Y-%m-%d")}` →
            `{train_config['selected_dates'][1].strftime("%Y-%m-%d")}`
            - **Afficher les métriques** : `{train_config['show_metrics']}`
            - **Afficher les prédictions** : `{train_config['show_preds']}`
            - **Afficher les résidus et la pente** : `{train_config['show_resid']}`
            - **Afficher l'interprétation** : `{train_config['show_interp']}`
            """)
        with col2:
            df_cols_exclues = pd.DataFrame({
                "Variables": available_columns,
                "Exclue": [
                    col in train_config["drop_cols"]
                    for col in available_columns
                ]
            })
            st.data_editor(
                df_cols_exclues,
                column_config={
                    "Exclue": st.column_config.CheckboxColumn("Exclue",
                                                              disabled=True)
                },
                hide_index=True,
                use_container_width=True,
                disabled=True,
                num_rows="dynamic"
            )
