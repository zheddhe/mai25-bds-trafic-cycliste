import streamlit as st
import pandas as pd
from smartcheck.dataframe_common import load_dataset_from_config
from smartcheck.modeling_project_specific import (
    train_timeseries_model,
    compute_metrics,
    plot_predictions,
    compute_residuals_plot,
    interpret_model,
    # render_shap_summary_streamlit,
)


# --- Chargement des données ML ready (en cache) ---
@st.cache_data
def run_cached_load_dataset_from_config():
    df = load_dataset_from_config("velo_comptage_ml_ready_data", sep=",", index_col=0)
    return df


# --- Entrainement du modèle (en cache) ---
@st.cache_data
def run_cached_train_timeseries_model(
    df: pd.DataFrame,
    model_type: str,
    target_col: str,
    drop_columns: list[str],
    use_ar1_ma24: bool,
    test_ratio: float,
):
    return train_timeseries_model(
        df,
        model_type,
        target_col=target_col,
        drop_columns=drop_columns,
        use_ar1_ma24=use_ar1_ma24,
        test_ratio=test_ratio,
    )


st.title("🧪 Évaluation des modèles")

st.markdown("""
Cette page vous permet de tester différents modèles de régression
sur les données de comptage vélo avec des options personnalisables :
* Type d'algorithme et répartition entre données d'entraînement et de test
* Données temporelles enrichies (AutoRegression de rang 1 et MoyenneMobile
24 précédentes valeurs)
* Choix des compteurs à modéliser (site + orientation)
* Visualisation in interprétation des prédictions par compteurs et analyse des
métriques et résidus associés
""")

with st.sidebar:
    sidebar_col1, sidebar_col2 = st.columns(2)
    with sidebar_col1:
        if st.button("🔁 Recharger le dataset"):
            run_cached_load_dataset_from_config.clear()  # type: ignore
            st.rerun()
    with sidebar_col2:
        if st.button("⚙️ Réentraîner le modèle"):
            run_cached_train_timeseries_model.clear()  # type: ignore
            st.rerun()


# --- Paramètres de modélisation ---
st.sidebar.header("🔧 Paramètres de modélisation")

# Type de modèle
algo_choice = st.sidebar.radio("Choisir l'algorithme", ("LinearRegression", "KNN"))

# Ratio train/test
split_ratio = st.sidebar.slider("Choisir la répartition Train/Test",
                                min_value=0.1, max_value=0.9,
                                value=0.75, step=0.05)

# Sélection des compteurs
available_counters = [
    ('Totem 73 boulevard de Sébastopol', 'S-N'),
    ('Totem 73 boulevard de Sébastopol', 'N-S'),
]
selected_sites = st.sidebar.multiselect(
    "Choisir les compteurs (site+orientation) à modéliser",
    options=available_counters,
    default=available_counters
)

# Colonnes temporelles
use_ar1_ma24 = st.sidebar.checkbox("Ajouter les variables explicatives AR(1)"
                                   "et Moy.Mobile(24)", value=True)

# --- Checkboxes pour visualisation ---
with st.sidebar:
    with st.expander("📊 Options du rapport de modélisation"):
        show_metrics = st.checkbox("Afficher métriques", value=True)
        show_predictions = st.checkbox("Afficher prédictions", value=True)
        show_residuals = st.checkbox("Afficher analyse des résidus", value=True)
        show_interpretation = st.checkbox("Afficher interprétabilité", value=False)
        # show_interpretation_shap = st.checkbox("Afficher interprétabilité SHAP",
        #                                        value=False)
        # background_method = st.selectbox(
        #     "Méthode background SHAP",
        #     options=["sample", "tail", "kmeans"]
        # )

# Filtrage des colonnes
with st.expander("📌 Colonnes à filtrer avant modélisation"):
    available_columns = [
        "weather_code_wmo_code",
        "date_et_heure_de_comptage_year",
        "date_et_heure_de_comptage_month",
        "date_et_heure_de_comptage_week",
        "date_et_heure_de_comptage_day",
        "date_et_heure_de_comptage_day_of_week",
        "date_et_heure_de_comptage_day_of_year",
        "date_et_heure_de_comptage_hour",
        "latitude",
        "longitude",
        "date_et_heure_de_comptage_cos_month",
        "date_et_heure_de_comptage_sin_month",
        "date_et_heure_de_comptage_sin_week",
        "weather_code_wmo_code",
    ]
    columns_to_drop = st.multiselect(
        "Colonnes à exclure du modèle",
        options=available_columns,
        default=available_columns,
        help="Ces colonnes seront supprimées du dataset avant entraînement du modèle."
    )

# --- Chargement des données ---
df_raw = run_cached_load_dataset_from_config()

if df_raw is not None and isinstance(df_raw, pd.DataFrame):
    st.success("✅ Données chargées avec succès.")
    # st.dataframe(df_raw.tail(100))

    grouped = df_raw.groupby(["nom_du_site_de_comptage", "orientation_compteur"])
    model_results = {}
    for compteur_id, df_compteur in grouped:
        if compteur_id in selected_sites:

            model_results[compteur_id] = run_cached_train_timeseries_model(
                df_compteur,
                algo_choice,
                "comptage_horaire",
                drop_columns=columns_to_drop,
                use_ar1_ma24=use_ar1_ma24,
                test_ratio=1-split_ratio,
            )

    if isinstance(model_results, dict) and selected_sites:

        compteur_labels = [
            f"{site} - {orientation}" for site, orientation in selected_sites
        ]
        tabs = st.tabs(compteur_labels)

        for tab, compteur_id in zip(tabs, selected_sites):
            with tab:
                results = model_results.get(compteur_id)

                if not results:
                    st.warning(f"Pas de résultat pour {compteur_id}")
                    continue

                if show_metrics:
                    st.markdown("### 📈 Métriques (données de test)")
                    metrics = compute_metrics(
                        results["y_test"],
                        results["y_test_pred"]
                    )
                    for key, value in metrics.items():
                        st.info(f"**{key}**: {value}")

                if show_predictions:
                    st.markdown("### 🔮 Prédictions (données de test)")
                    fig_pred = plot_predictions(
                        compteur=str(compteur_id),
                        dates=results["X_test_dates"],
                        y_true=results["y_test"],
                        y_pred=results["y_test_pred"],
                        periode_limite=('2025-04-01', '2025-04-16')
                    )
                    st.pyplot(fig_pred)

                if show_residuals:
                    st.markdown("### 🧾 Analyse des résidus (données de test)")
                    fig1_res, fig2_res, pente = compute_residuals_plot(
                        compteur=str(compteur_id),
                        dates=results["X_test_dates"],
                        y_true=results["y_test"],
                        y_pred=results["y_test_pred"],
                        periode_limite=('2025-04-01', '2025-04-16')
                    )
                    st.pyplot(fig1_res)
                    st.info("coefficient de la dérive des résidus"
                            f"(pente de la régression linéaire): {pente:.4f}")
                    st.pyplot(fig2_res)

                if show_interpretation:
                    st.markdown("### 🧠 Interprétabilité (données de test)")
                    fig_interp_list = interpret_model(str(compteur_id), results)
                    if fig_interp_list is not None:
                        for fig_interp in fig_interp_list:
                            st.pyplot(fig_interp)

                # if show_interpretation_shap:
                #     st.markdown("### 🧠 Interprétabilité SHAP (données de train)")
                #     render_shap_summary_streamlit(
                #         pipe=results["pipe"],
                #         X=results["X_train"],
                #         background_method=background_method,
                #         nb_samples=50,
                #         background_size=100,
                #         max_display=15
                #     )

else:
    st.error("❌ Impossible de charger les données.")
