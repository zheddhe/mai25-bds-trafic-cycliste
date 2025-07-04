import streamlit as st
from smartcheck.dataframe_common import load_dataset_from_config
import pandas as pd

st.title("🔍 Exploration des données")


# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_refactored_data"

# --- Chargement des données ---
@st.cache_data
def cached_load_dataset_exploration():
    df = load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)
    return df


with st.sidebar:
    if st.button("🔁 Recharger données", key="reload_button"):
        cached_load_dataset_exploration.clear()  # type: ignore
        st.rerun()

df_raw = cached_load_dataset_exploration()

if df_raw is not None and isinstance(df_raw, pd.DataFrame):
    st.success("✅ Données chargées avec succès.")
    st.dataframe(df_raw.tail(100))

else:
    st.error("❌ Impossible de charger les données.")
