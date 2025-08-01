import streamlit as st
from app.utils.model_logic import (
    run_evaluation_per_compteur,
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

# --- Enrichissement du menu ---
with st.sidebar:
    df = manage_dataset_modeling(st)
    train_config = manage_sidebar_modeling_parameters(st)

display_train_parameters(train_config, st)

# --- Controles Generaux avant entrainement et rendu
if not train_config["selected_sites"]:
    st.warning("Sélectionnez au moins un compteur à modéliser.")
    st.stop()
if train_config["range"][0] == train_config["range"][1]:
    st.warning("La plage sélectionnée est vide.")
    st.stop()

# --- Entrainement des compteurs sélectionnés
with st.spinner("⏳ Entraînement des modèles en cours..."):
    results, metrics_table = manage_training(df, train_config)

# --- Synthèse globale des performances ---
if metrics_table:
    st.markdown("## 🧾 Synthèse des métriques de modélisation")
    display_metrics_table(metrics_table, st_module=st)

# --- Rapports par compteur ---
st.markdown("## 🎯 Rapports par compteur")
run_evaluation_per_compteur(results, train_config, st_module=st)
