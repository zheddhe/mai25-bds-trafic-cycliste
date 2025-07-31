import streamlit as st
from app.utils.dataexpl_logic import (
    manage_dataset_exploration,
    display_duplicate_and_discontinuities,
    display_counting_regrouped_stat,
)

st.title("🔍 Exploration statistique intéractive des données")
st.markdown("""
Cette section vous permet d'explorer les données afin d'observer
différents types de statistiques mesurant la qualité et la pertinence des données de
comptage vélo avec des paramètres de regroupement personnalisables.
""")
st.info("""
👈 Le dataset est préchargé mais vous pouvez forcer son rechargement via le menu
""")

with st.sidebar:
    df_raw = manage_dataset_exploration(st)

with st.expander("🚨 Identification des doublons et des discontinuités"
                 "dans les périodes de relevé horaire", expanded=True):
    display_duplicate_and_discontinuities(df_raw, st)

with st.expander("📊 Statistiques descriptives globales", expanded=True):
    st.dataframe(df_raw.describe(include='all').T)

with st.expander("🧮 Statistiques numériques descriptives après regroupements autour de "
                 "la variable cible (comptage horaire)", expanded=True):
    display_counting_regrouped_stat(df_raw, st)
