import streamlit as st
import pandas as pd
from smartcheck.dataframe_common import load_dataset_from_config
from smartcheck.modeling_project_specific import (
    train_timeseries_model,
    compute_metrics,
)
from app.utils.model_logic import (
    run_evaluation_per_compteur,
    display_metrics_table,
    get_selected_period,
)
import os


@st.cache_data(show_spinner=True)
def cached_load_dataset_ml():
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame(columns=["nom_du_site_de_comptage",
                                     "orientation_compteur", "comptage_horaire"])
    return load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)


@st.cache_data(show_spinner=True)
def cached_train_model(df,
                       model_type,
                       scaler_type,
                       target_col,
                       drop_cols,
                       temp_feats,
                       test_ratio):
    if os.environ.get("IS_TESTING") == "1":
        return {
            "y_test": [1, 2],
            "y_test_pred": [1.1, 1.9],
            "X_test_dates": pd.date_range("2025-04-01", periods=2, freq="h")
        }
    return train_timeseries_model(
        df,
        model_type,
        scaler_type,
        target_col=target_col,
        drop_columns=drop_cols,
        temp_feats=temp_feats,
        test_ratio=test_ratio,
    )


# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_ml_ready_data"
DEFAULT_TEST_PERIOD = ('2025-04-01', '2025-04-16')
MAX_TEST_PERIOD = ('2025-01-02', '2025-04-16')
SITE_LABELS = {
    ('Totem 73 boulevard de Sébastopol', 'S-N'): "Sébastopol - S-N",
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol - N-S",
    ('102 boulevard de Magenta', 'SE-NO'): "Magenta - O-E",
    ('Pont de Bercy', 'NE-SO'): "Bercy - NE-SO",
    ('Pont de Bercy', 'NE-SO'): "Bercy - NE-SO",
    ('135 avenue Daumesnil', 'SE-NO'): "Daumesnil - SE-NO",
    ("180 avenue d'Italie", 'N-S'): "Italie - N-S",
    ('27 quai de la Tournelle', 'NO-SE'): "Tournelle - NO-SE",
    ('27 quai de la Tournelle', 'SE-NO'): "Tournelle - SE-NO",
}
SITE_LABELS_DEFAULT = {
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol - N-S",
    ('102 boulevard de Magenta', 'SE-NO'): "Magenta - O-E",
}
EXCLUDED_COLUMNS_DEFAULT = [
    "weather_code_wmo_code",
    "date_et_heure_de_comptage_year",
    "date_et_heure_de_comptage_month",
    "date_et_heure_de_comptage_day",
    "date_et_heure_de_comptage_day_of_year",
    "date_et_heure_de_comptage_day_of_week",
    "date_et_heure_de_comptage_hour",
    "date_et_heure_de_comptage_week",
    "latitude",
    "longitude",
    "date_et_heure_de_comptage_cos_month",
    "date_et_heure_de_comptage_sin_month",
    "date_et_heure_de_comptage_sin_week",
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
    "date_et_heure_de_comptage_cos_week",
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
]

# --- UI Setup ---
st.title("🧪 Évaluation des modèles")
st.markdown("""
Cette page vous permet de tester différents modèles de régression
sur les données de comptage vélo avec des options personnalisables.
""")

with st.sidebar:
    col_b1, col_b2 = st.columns(2)
    if col_b1.button("🔁 Dataset"):
        cached_load_dataset_ml.clear()  # type: ignore
        st.rerun()
    if col_b2.button("⚙️ Model"):
        cached_train_model.clear()  # type: ignore
        st.rerun()

    st.header("🔧 Paramètres")

    algo = st.radio("Algorithme", AVAILABLE_MODELS)

    ar_nb = st.slider("Nb d'Auto-Régression", 0, 7, 0, 1)
    mm_nb = st.slider("Nb de Moyennes Mobiles", 0, 7, 0, 1)
    mm_season = st.number_input("Taille de la fenêtre mobile (1 lag = 1 heure)",
                                min_value=2, max_value=24*7, value=24)

    scaler = st.radio("Mise à l'échelle",
                      ("MinMaxScaler", "StandardScaler", "RobustScaler"))

    split = st.slider("Répartition Train/Test", 0.1, 0.9, 0.75, 0.05)

    with st.expander("📊 Option des rapports") as st_report_opt:
        selected_dates = get_selected_period(
            default_start=DEFAULT_TEST_PERIOD[0],
            default_end=DEFAULT_TEST_PERIOD[1],
            min_dt_str=MAX_TEST_PERIOD[0],
            max_dt_str=MAX_TEST_PERIOD[1],
            label="📆 Période de test à représenter",
            st_module=st_report_opt
        )
        show_metrics = st.checkbox("Afficher métriques", value=True)
        show_preds = st.checkbox("Afficher prédictions", value=True)
        show_resid = st.checkbox("Afficher résidus", value=False)
        show_interp = st.checkbox("Afficher interprétation", value=False)

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
        )
        drop_cols = edited_df.loc[edited_df["Exclue"], "Variable"].tolist()

# --- Chargement des données ---
with st.spinner("⏳ Chargement du dataset en cours..."):
    df = cached_load_dataset_ml()
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("❌ Erreur lors du chargement des données.")
        st.stop()

st.success(f"✅ Données [{DATASET_NAME}] chargées avec succès.")
grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])

# --- Sélection des compteurs ---
label_options = list(SITE_LABELS.values())
label_default_options = list(SITE_LABELS_DEFAULT.values())
selected_labels = st.multiselect(
    "🎯 Compteurs à modéliser",
    label_options,
    default=label_default_options
)
selected_sites = [k for k, v in SITE_LABELS.items() if v in selected_labels]

with st.spinner("⏳ Entraînement des modèles en cours..."):
    results = {}
    metrics_table = []
    for compteur_id, df_site in grouped:
        if compteur_id in selected_sites:
            res = cached_train_model(
                df_site,
                algo,
                scaler,
                "comptage_horaire",
                drop_cols,
                [ar_nb, mm_nb, mm_season],
                1 - split,
            )
            results[compteur_id] = res
            if show_metrics:
                train_metrics = compute_metrics(res["y_train"], res["y_train_pred"])
                test_metrics = compute_metrics(res["y_test"], res["y_test_pred"])
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

if not results:
    st.warning("Aucun compteur sélectionné.")
    st.stop()

# --- Synthèse globale des performances ---
if show_metrics and metrics_table:
    st.markdown("## 🧾 Synthèse des métriques de modélisation par compteur")
    display_metrics_table(metrics_table, st_module=st)

# --- Affichage par compteur ---
run_evaluation_per_compteur(
    results, SITE_LABELS,
    show_metrics, show_preds,
    show_resid, show_interp,
    periode_limite=selected_dates,
    st_module=st
)
