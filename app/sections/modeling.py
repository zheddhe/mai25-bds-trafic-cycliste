import streamlit as st
import pandas as pd
from smartcheck.dataframe_common import (
    load_dataset_from_config,
    apply_percent_range_selection,
)
from smartcheck.modeling_project_specific import (
    train_timeseries_model,
    compute_metrics,
)
from app.utils.model_logic import (
    run_evaluation_per_compteur,
    display_metrics_table,
    # get_selected_period,
    display_train_parameters,
)
import os


@st.cache_data(show_spinner=True)
def cached_load_dataset_ml():
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame(columns=["nom_du_site_de_comptage",
                                     "orientation_compteur", "comptage_horaire"])
    return load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)


@st.cache_data(show_spinner=True)
def cached_train_model(df_compteur,
                       model_type,
                       scaler_type,
                       target_col,
                       drop_columns,
                       temp_feats,
                       test_ratio,
                       forecast):
    if os.environ.get("IS_TESTING") == "1":
        return {
            "y_test": [1, 2],
            "y_test_pred": [1.1, 1.9],
            "X_test_dates": pd.date_range("2025-04-01", periods=2, freq="h")
        }
    return train_timeseries_model(
        df_compteur=df_compteur,
        model_type=model_type,
        scaler_type=scaler_type,
        target_col=target_col,
        drop_columns=drop_columns,
        temp_feats=temp_feats,
        test_ratio=test_ratio,
        forecast=forecast,
    )


# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_ml_ready_data"
DEFAULT_TEST_PERIOD = ('2025-04-01', '2025-04-14')
MAX_TEST_PERIOD = ('2024-03-01', '2025-04-14')
SITE_LABELS = {
    ('Totem 73 boulevard de Sébastopol', 'S-N'): "Sébastopol_S-N",
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol_N-S",
    ('102 boulevard de Magenta', 'SE-NO'): "Magenta_SE-NO",
    ('Pont de Bercy', 'NE-SO'): "Bercy_NE-SO",
    ('Pont de Bercy', 'SO-NE'): "Bercy_SO-NE",
    ('135 avenue Daumesnil', 'SE-NO'): "Daumesnil_SE-NO",
    ("180 avenue d'Italie", 'N-S'): "Italie_N-S",
    ('27 quai de la Tournelle', 'NO-SE'): "Tournelle_NO-SE",
    ('27 quai de la Tournelle', 'SE-NO'): "Tournelle_SE-NO",
}
SITE_LABELS_DEFAULT = {
    ('Totem 73 boulevard de Sébastopol', 'S-N'): "Sébastopol_S-N",
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol_N-S",
    ('102 boulevard de Magenta', 'SE-NO'): "Magenta_SE-NO",
    ('Pont de Bercy', 'NE-SO'): "Bercy_NE-SO",
    ('Pont de Bercy', 'SO-NE'): "Bercy_SO-NE",
    ('135 avenue Daumesnil', 'SE-NO'): "Daumesnil_SE-NO",
    ("180 avenue d'Italie", 'N-S'): "Italie_N-S",
    ('27 quai de la Tournelle', 'NO-SE'): "Tournelle_NO-SE",
    ('27 quai de la Tournelle', 'SE-NO'): "Tournelle_SE-NO",
}
EXCLUDED_COLUMNS_DEFAULT = [
    "weather_code_wmo_code",
    "date_et_heure_de_comptage_hour",
    "date_et_heure_de_comptage_day",
    "date_et_heure_de_comptage_day_of_year",
    "date_et_heure_de_comptage_day_of_week",
    "date_et_heure_de_comptage_week",
    "date_et_heure_de_comptage_month",
    "date_et_heure_de_comptage_year",
    "latitude",
    "longitude",
]
AVAILABLE_COLUMNS_TO_EXCLUDE = [
    "weather_code_wmo_code_category",
    "arrondissement",
    "jour_ferie",
    "vacances_scolaires",
    "temperature_2m_c",
    "rain_mm",
    "snowfall_cm",
    "elevation",
    "date_et_heure_de_comptage_week_end",
    "date_et_heure_de_comptage_sin_hour",
    "date_et_heure_de_comptage_cos_hour",
    "date_et_heure_de_comptage_sin_day_of_week",
    "date_et_heure_de_comptage_cos_day_of_week",
    "date_et_heure_de_comptage_sin_week",
    "date_et_heure_de_comptage_cos_week",
    "date_et_heure_de_comptage_cos_month",
    "date_et_heure_de_comptage_sin_month",
    "date_et_heure_de_comptage_cos_day_of_year",
    "date_et_heure_de_comptage_sin_day_of_year",
]
AVAILABLE_COLUMNS = EXCLUDED_COLUMNS_DEFAULT+AVAILABLE_COLUMNS_TO_EXCLUDE
MANDATORY_COLUMNS = [
    "nom_du_site_de_comptage",
    "orientation_compteur",
    "date_et_heure_de_comptage_local",
    "comptage_horaire",
]
AVAILABLE_MODELS = [
    "LinearRegression",
    "KNN",
    "RandomForest",
    "XGBoost",
    "ElasticNet (*)",
]

st.title("🧪 Évaluation des modèles")
st.markdown("""
Cette page vous permet de tester différents modèles de régression
sur les données de comptage vélo avec des options personnalisables.
> - Le dataset est préchargé mais vous pouvez forcer son rechargement depuis google
drive
""")

# --- Chargement des données ---
with st.spinner("⏳ Chargement du dataset en cours..."):
    df = cached_load_dataset_ml()
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("❌ Erreur lors du chargement des données.")
        st.stop()

st.success(f"✅ Données [{DATASET_NAME}] chargées avec succès.")
grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])

# --- Enrichissement du menu ---
with st.sidebar:

    # --- Bouton de rechargement des données de comptage ---
    if st.button("🔁 Rechargement du Dataset"):
        cached_load_dataset_ml.clear()  # type: ignore
        st.rerun()

    st.header("🔧 Paramètres")

    # --- Sélection des compteurs ---
    label_options = list(SITE_LABELS.values())
    label_default_options = list(SITE_LABELS_DEFAULT.values())
    selected_labels = st.multiselect(
        "🎯 Compteurs à modéliser",
        label_options,
        default=label_default_options
    )
    selected_sites = [k for k, v in SITE_LABELS.items() if v in selected_labels]

    # --- Sélection des modèles ---
    model = st.radio("Modèle", AVAILABLE_MODELS, key="model_rad")

    # --- Sélection des paramètres d'auto regression/moyenne mobile ---
    ar_nb = st.slider("Nb d'Auto-Régression", 0, 7, 0, 1, key="ar_nb_sld")
    mm_nb = st.slider("Nb de Moyennes Mobiles", 0, 7, 0, 1, key="ar_mm_nb")
    mm_season = st.number_input("Taille de la fenêtre mobile (1 lag = 1 heure)",
                                min_value=2, max_value=24*7,
                                value=24, key="mm_season_inp")
    use_forecast = st.checkbox("Prédiction dynamique des AR/MM (**)",
                               value=False,
                               key="use_forecast_cb")

    # --- Sélection du scaler ---
    scaler = st.radio("Mise à l'échelle",
                      ("MinMaxScaler", "StandardScaler", "RobustScaler"),
                      key="scaler_rad")

    # --- Echantillonnage du dataset + répartition train/test ---
    range = st.slider("Prendre une portion du dataset d'origine", 0.0, 100.0,
                      (0.0, 100.0), 0.1, format="%.1f %%", key="range_sld")
    split = st.slider("Répartition Train/Test", 0.1, 0.9, 0.75, 0.05,
                      key="split_sld")

    # --- Explications
    st.markdown("""
> (*) Entrainement **_couteux_** avec recherche Bayesienne d'hyperparamètres
>
> (**) Prédiction récursive **_couteuse_** avec ré-infusion des AR/MM recalculés sur la
base des données prédites plutôt que réelles
""")
    # --- Sélection des variables explicatives ---
    with st.expander("❌ **Variables explicatives à exclure**"):
        df_checkbox = pd.DataFrame({
            "Variable": AVAILABLE_COLUMNS,
            "Exclue": [
                col in EXCLUDED_COLUMNS_DEFAULT
                for col in AVAILABLE_COLUMNS
            ]
        })
        st.markdown("🔒 Les colonnes suivantes sont obligatoires:")
        st.code("\n".join(MANDATORY_COLUMNS), language="markdown")
        edited_df = st.data_editor(
            df_checkbox,
            column_config={
                "Exclue": st.column_config.CheckboxColumn("Exclue")
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="edited_df_de"
        )
        drop_cols = edited_df.loc[edited_df["Exclue"], "Variable"].tolist()

    # --- Option des rapports ---
    with st.expander("📊 Option des rapports") as st_report_opt:
        col_date_min, col_date_max = st.columns(2)
        with col_date_min:
            start_date = st.date_input("Début de l'affichage",
                                       value=DEFAULT_TEST_PERIOD[0],
                                       min_value=MAX_TEST_PERIOD[0],
                                       max_value=MAX_TEST_PERIOD[1])
        with col_date_max:
            end_date = st.date_input("Fin de l'affichage",
                                     value=DEFAULT_TEST_PERIOD[1],
                                     min_value=MAX_TEST_PERIOD[0],
                                     max_value=MAX_TEST_PERIOD[1])
        selected_dates = (start_date, end_date)
        show_metrics = st.checkbox("Afficher métriques", value=True,
                                   key="show_metrics_cb")
        show_preds = st.checkbox("Afficher prédictions", value=True,
                                 key="show_preds_cb")
        show_resid = st.checkbox("Afficher résidus", value=True,
                                 key="show_resid_cb")
        show_interp = st.checkbox("Afficher interprétation", value=True,
                                  key="show_interp_cb")

train_config = {
    "model": model,
    "scaler": scaler,
    "ar_nb": ar_nb,
    "mm_nb": mm_nb,
    "mm_season": mm_season,
    "use_forecast": use_forecast,
    "range": range,
    "split": split,
    "drop_cols": drop_cols.copy(),
    "selected_dates": selected_dates,
    "show_metrics": show_metrics,
    "show_preds": show_preds,
    "show_resid": show_resid,
    "show_interp": show_interp,
}
display_train_parameters(train_config, AVAILABLE_COLUMNS, st)

# --- Controles Generaux avant entrainement et rendu
if not selected_sites:
    st.warning("Sélectionnez au moins un compteur à modéliser.")
    st.stop()
if range[0] == range[1]:
    st.warning("La plage sélectionnée est vide.")
    st.stop()

with st.spinner("⏳ Entraînement des modèles en cours..."):
    results = {}
    metrics_table = []
    for compteur_id, df_compteur in grouped:
        if compteur_id in selected_sites:
            res = cached_train_model(
                df_compteur=apply_percent_range_selection(
                    df_compteur,
                    range,
                ),
                model_type=model,
                scaler_type=scaler,
                target_col="comptage_horaire",
                drop_columns=drop_cols,
                temp_feats=[ar_nb, mm_nb, mm_season],
                test_ratio=1 - split,
                forecast=use_forecast,
            )
            results[compteur_id] = res
            train_metrics = compute_metrics(res["y_train"],
                                            res["y_train_pred"])
            test_metrics = compute_metrics(res["y_test"],
                                           res["y_test_pred"])
            combined_row = {
                "compteur": SITE_LABELS[compteur_id],
                "description": compteur_id,
                "R2_train": train_metrics.get("R2", None),
                "RMSE_train": train_metrics.get("RMSE", None),
                "MAE_train": train_metrics.get("MAE", None),
                "R2_test": test_metrics.get("R2", None),
                "RMSE_test": test_metrics.get("RMSE", None),
                "MAE_test": test_metrics.get("MAE", None),
            }
            metrics_table.append(combined_row)

# --- Synthèse globale des performances ---
if train_config and train_config["show_metrics"] and metrics_table:
    st.markdown("## 🧾 Synthèse des métriques de modélisation par compteur")
    display_metrics_table(metrics_table, st_module=st)
    # --- Affichage par compteur ---
    run_evaluation_per_compteur(
        results,
        SITE_LABELS,
        train_config["show_metrics"],
        train_config["show_preds"],
        train_config["show_resid"],
        train_config["show_interp"],
        periode_limite=train_config["selected_dates"],
        st_module=st
    )
