import streamlit as st
from smartcheck.dataframe_common import load_dataset_from_config
import pandas as pd

st.title("🔍 Exploration des données")


# --- Chargement des données ---
@st.cache_data
def load_data_exploration():
    df = load_dataset_from_config("velo_comptage_refactored_data", sep=",", index_col=0)
    return df


with st.sidebar:
    if st.button("🔁 Recharger les données", key="reload_button"):
        load_data_exploration.clear()  # type: ignore
        st.rerun()

df_raw = load_data_exploration()

if df_raw is not None and isinstance(df_raw, pd.DataFrame):
    st.success("✅ Données chargées avec succès.")
    st.dataframe(df_raw.tail(100))

else:
    st.error("❌ Impossible de charger les données.")
