import streamlit as st
import pandas as pd
from smartcheck.dataframe_common import load_dataset_from_config
from smartcheck.modeling_project_specific import (
    train_timeseries_model,
    compute_metrics,
    plot_predictions,
    compute_residuals_plot,
    interpret_model,
)

# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_ml_ready_data"
DEFAULT_PERIOD = ('2025-04-01', '2025-04-16')

SITE_LABELS = {
    ('Totem 73 boulevard de Sébastopol', 'S-N'): "Sébastopol - S↑N",
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol - N↓S",
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
    return load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)

@st.cache_data(show_spinner=True)
def cached_train_model(df, model_type, target_col, drop_cols,
                       use_ar_ma, test_ratio):
    return train_timeseries_model(
        df,
        model_type,
        target_col=target_col,
        drop_columns=drop_cols,
        use_ar1_ma24=use_ar_ma,
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
    algo = st.radio("Algorithme", ("LinearRegression", "KNN"))
    split = st.slider("Équilibre Train/Test", 0.1, 0.9, 0.75, 0.05)
    use_ar_ma = st.checkbox("Ajout AR(1)+MA(24)", value=True)

    label_options = list(SITE_LABELS.values())
    selected_labels = st.multiselect("Compteurs ", label_options, default=label_options)
    selected_sites = [k for k, v in SITE_LABELS.items() if v in selected_labels]

    drop_cols = st.multiselect("❌ Colonnes à exclure",
                                options=AVAILABLE_COLUMNS,
                                default=AVAILABLE_COLUMNS)

    with st.expander("📊 Rapport"):
        show_metrics = st.checkbox("Afficher métriques", value=True)
        show_preds = st.checkbox("Afficher prédictions", value=True)
        show_resid = st.checkbox("Afficher résidus", value=True)
        show_interp = st.checkbox("Afficher interprétation", value=False)

    col1, col2 = st.columns(2)
    if col1.button("🔁 Recharger les données"):
        cached_load_dataset_ml.clear()  # type: ignore
        st.rerun()
    if col2.button("⚙️ Réentraîner le modèle"):
        cached_train_model.clear()  # type: ignore
        st.rerun()

# --- Chargement des données ---
df = cached_load_dataset_ml()
if df is None or not isinstance(df, pd.DataFrame):
    st.error("❌ Erreur lors du chargement des données.")
    st.stop()

st.success("✅ Données chargées avec succès.")
grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])

results = {}
metrics_table = []

with st.spinner("⏳ Entraînement en cours..."):
    for compteur_id, df_site in grouped:
        if compteur_id in selected_sites:
            res = cached_train_model(
                df_site,
                algo,
                "comptage_horaire",
                drop_cols,
                use_ar_ma,
                1 - split,
            )
            results[compteur_id] = res
            if show_metrics:
                metrics = compute_metrics(res["y_test"], res["y_test_pred"])
                metrics_table.append({"compteur": SITE_LABELS[compteur_id], **metrics})

if not results:
    st.warning("Aucun compteur sélectionné.")
    st.stop()

# --- Synthèse globale des performances ---
if show_metrics and metrics_table:
    st.markdown("## 🧾 Synthèse des métriques par compteur")
    df_metrics = pd.DataFrame(metrics_table)
    st.dataframe(df_metrics.set_index("compteur"))

# --- Affichage par compteur ---
for compteur_id, res in results.items():
    label = SITE_LABELS[compteur_id]
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
                periode_limite=DEFAULT_PERIOD,
            )
            st.pyplot(fig)

        if show_resid:
            st.markdown("### 🧾 Résidus")
            fig1, fig2, slope = compute_residuals_plot(
                compteur=label,
                dates=res["X_test_dates"],
                y_true=res["y_test"],
                y_pred=res["y_test_pred"],
                periode_limite=DEFAULT_PERIOD,
            )
            st.pyplot(fig1)
            st.info(f"Dérive des résidus : pente = {slope:.4f}")
            st.pyplot(fig2)

        if show_interp:
            st.markdown("### 🧠 Interprétation")
            interp_figs = interpret_model(label, res)
            if interp_figs:
                for fig in interp_figs:
                    st.pyplot(fig)
