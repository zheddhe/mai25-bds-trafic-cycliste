import streamlit as st
from smartcheck.dataframe_common import load_dataset_from_config
import pandas as pd

st.title("🔍 Exploration des données")

df = load_dataset_from_config("velo_comptage_refactored_data", sep=",")
if df is not None and isinstance(df, pd.DataFrame):
    st.success("Dataset chargé avec succès.")
    st.dataframe(df.head(100))
else:
    st.error("Impossible de charger les données.")
