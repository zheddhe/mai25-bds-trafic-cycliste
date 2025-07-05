import streamlit as st
import pandas as pd
from smartcheck.dataframe_common import load_dataset_from_config
from smartcheck.modeling_project_specific import (
    train_timeseries_model,
    compute_metrics,
)
from app.utils.model_logic import run_evaluation_per_compteur
import os

# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_ml_ready_data"
DEFAULT_PERIOD = ('2025-04-01', '2025-04-16')
SITE_LABELS = {
    ('Totem 73 boulevard de Sébastopol', 'S-N'): "Sébastopol - S-N",
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol - N-S",
    ('Totem 64 Rue de Rivoli', 'O-E'): "Rivoli - O-E",
    ('Pont de Bercy', 'NE-SO'): "Bercy - NE-SO",
    ('Pont de Bercy', 'NE-SO'): "Bercy - NE-SO",
    ('135 avenue Daumesnil', 'SE-NO'): "Daumesnil - SE-NO",
    ("180 avenue d'Italie", 'N-S'): "Italie - N-S",
    ('27 quai de la Tournelle', 'NO-SE'): "Tournelle - NO-SE",
    ('27 quai de la Tournelle', 'SE-NO'): "Tournelle - SE-NO",
}
AVAILABLE_COLUMNS = sorted([
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
])


@st.cache_data(show_spinner=True)
def cached_load_dataset_ml():
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame(columns=["nom_du_site_de_comptage",
                                     "orientation_compteur", "comptage_horaire"])
    return load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)


@st.cache_data(show_spinner=True)
def cached_train_model(df, model_type, scaler_type, target_col,
                       drop_cols, temp_feats, test_ratio):
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


# --- UI Setup ---
st.title("🧪 Évaluation des modèles")
st.markdown("""
Cette page vous permet de tester différents modèles de régression
sur les données de comptage vélo avec des options personnalisables.
""")

with st.sidebar:
    st.header("🔧 Paramètres")
    algo = st.radio("Algorithme", ("LinearRegression", "KNN", "RandomForest"))
    temp_feats = st.radio("Variables temporelles additionnelles",
                          ("Aucune", "AR(1) et MM(24)"))
    scaler = st.radio("Mise à l'échelle",
                      ("MinMaxScaler", "StandardScaler", "RobustScaler"))
    split = st.slider("Répartition Train/Test", 0.1, 0.9, 0.75, 0.05)

    with st.expander("📊 Rapport"):
        show_metrics = st.checkbox("Afficher métriques", value=True)
        show_preds = st.checkbox("Afficher prédictions", value=True)
        show_resid = st.checkbox("Afficher résidus", value=False)
        show_interp = st.checkbox("Afficher interprétation", value=False)

    col1, col2 = st.columns(2)
    if col1.button("🔁 Recharger les données"):
        cached_load_dataset_ml.clear()  # type: ignore
        st.rerun()
    if col2.button("⚙️ Réentraîner le modèle"):
        cached_train_model.clear()  # type: ignore
        st.rerun()

# --- Chargement des données ---
with st.spinner("⏳ Chargement en cours..."):
    df = cached_load_dataset_ml()
    if df is None or not isinstance(df, pd.DataFrame):
        st.error("❌ Erreur lors du chargement des données.")
        st.stop()

st.success("✅ Données chargées avec succès.")
grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])

# --- Options de filtrage ---
label_options = list(SITE_LABELS.values())
selected_labels = st.multiselect("🎯 Compteurs à modéliser",
                                 label_options, default=label_options)
selected_sites = [k for k, v in SITE_LABELS.items() if v in selected_labels]

drop_cols = st.multiselect(
    "❌ Colonnes à exclure",
    options=AVAILABLE_COLUMNS,
    default=AVAILABLE_COLUMNS
)

results = {}
metrics_table = []

with st.spinner("⏳ Entraînement en cours..."):
    for compteur_id, df_site in grouped:
        if compteur_id in selected_sites:
            res = cached_train_model(
                df_site,
                algo,
                scaler,
                "comptage_horaire",
                drop_cols,
                temp_feats,
                1 - split,
            )
            results[compteur_id] = res
            if show_metrics:
                metrics = compute_metrics(res["y_test"], res["y_test_pred"])
                metrics_table.append({"compteur": SITE_LABELS[compteur_id],
                                      **{'description': compteur_id},
                                      **metrics})

if not results:
    st.warning("Aucun compteur sélectionné.")
    st.stop()

# --- Synthèse globale des performances ---
if show_metrics and metrics_table:
    st.markdown("## 🧾 Synthèse des métriques de modélisation par compteur")
    df_metrics = pd.DataFrame(metrics_table)
    st.dataframe(df_metrics.set_index("compteur"))

# --- Affichage par compteur ---
run_evaluation_per_compteur(
    results, SITE_LABELS,
    show_metrics, show_preds,
    show_resid, show_interp,
    periode_limite=DEFAULT_PERIOD,
    st_module=st
)
