import streamlit as st
from app.utils.model_logic import (
    display_report_per_counter,
    display_metrics_table,
    manage_dataset_modeling,
    manage_sidebar_modeling_parameters,
    display_train_parameters,
    manage_training,
)

st.title("🧪 Laboratoire d'évaluation des modèles")
st.markdown("""
Cette section vous permet de tester différents modèles de régression
sur les données de comptage vélo avec des options personnalisables.
""")
st.info("""
👈 Le dataset est préchargé mais vous pouvez forcer son rechargement via le menu
""")

with st.sidebar:
    # --- Enrichissement du menu ---
    df = manage_dataset_modeling(st)
    train_config = manage_sidebar_modeling_parameters(st)

display_train_parameters(train_config, st)

# --- Entrainement des compteurs sélectionnés
with st.spinner("⏳ Entraînement des modèles en cours..."):
    results, metrics_table = manage_training(train_config, df)

# --- Synthèse globale des performances ---
st.markdown("## 🧾 Synthèse des métriques de modélisation")
display_metrics_table(metrics_table, st_module=st)

# --- Rapports par compteur ---
st.markdown("## 🎯 Rapports par compteur")
with st.spinner("⏳ Construction des rapports en cours..."):
    display_report_per_counter(results, train_config, st_module=st)
