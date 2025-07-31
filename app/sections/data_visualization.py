import streamlit as st
from app.utils.dataviz_logic import (
    manage_dataset_visualization,
    display_distrib_counting_graphics,
    display_average_counting_graphics,
    display_total_counting_graphics,
    display_top_sites_graphics,
    display_distrib_multi_level_graphics,
    display_sites_map_graphics,
)


st.title("📈 Visualisations intéractives des données")
st.markdown("""
Cette section vous permet de plonger dans différentes visualisation ces données de
comptage vélo avec des paramètres personnalisables et des graphiques interactifs.
""")
st.info("""
👈 Le dataset est préchargé mais vous pouvez forcer son rechargement via le menu
""")

with st.sidebar:
    df_raw = manage_dataset_visualization(st)

display_distrib_counting_graphics(df_raw, st)

display_average_counting_graphics(df_raw, st)

display_total_counting_graphics(df_raw, st)

display_top_sites_graphics(df_raw, st)

display_distrib_multi_level_graphics(df_raw, st)

display_sites_map_graphics(df_raw, st)
