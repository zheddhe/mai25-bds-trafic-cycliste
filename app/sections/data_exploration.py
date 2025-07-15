import streamlit as st
from smartcheck.dataframe_common import load_dataset_from_config
import pandas as pd
import os


# --- Chargement des données ---
@st.cache_data
def cached_load_dataset_exploration():
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame(columns=["nom_du_site_de_comptage",
                                     "orientation_compteur", "comptage_horaire"])
    df = load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)
    return df


# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_refactored_data"

st.title("🔍 Exploration des données")

with st.sidebar:
    if st.button("🔁 Dataset", key="reload_button"):
        cached_load_dataset_exploration.clear()  # type: ignore
        st.rerun()

with st.spinner("⏳ Chargement du dataset en cours..."):
    df_raw = cached_load_dataset_exploration()

if df_raw is not None and isinstance(df_raw, pd.DataFrame):
    st.success(f"✅ Données [{DATASET_NAME}] chargées avec succès.")
    st.dataframe(df_raw.tail())

    with st.expander("📊 Statistiques descriptives globales"):
        st.dataframe(df_raw.describe(include='all').T)

    with st.expander("🧮 Statistique par regroupement autour de "
                     "la variable cible (comptage horaire)"):
        group_column = st.multiselect(
            "📌 Groupe d’agrégation",
            [
                "nom_du_site_de_comptage",
                "orientation_compteur",
                "date_et_heure_de_comptage_hour",
                "date_et_heure_de_comptage_day_of_week",
                "arrondissement",
            ],
            default=[
                "nom_du_site_de_comptage",
                "orientation_compteur",
            ],
        )

        if group_column:
            grouped_stats = df_raw.groupby(group_column)[
                ["comptage_horaire"]
            ].agg(["count", "sum", "mean", "std"])
            df_stats = pd.DataFrame(grouped_stats)
            styled_df = (
                df_stats.style
                .background_gradient(
                    subset=pd.IndexSlice[:, (
                        'comptage_horaire',
                        'count'
                    )],  # type: ignore
                    vmin=0,
                    cmap="RdYlGn",  # green = good, red = bad
                )
                .background_gradient(
                    subset=pd.IndexSlice[:, (
                        'comptage_horaire',
                        'sum'
                    )],  # type: ignore
                    cmap="coolwarm",
                )
            )
            st.dataframe(styled_df, use_container_width=True)

    with st.expander("🚨 Doublons et valeurs manquantes"):
        st.write(f"Nombre de lignes dupliquées : {df_raw.duplicated().sum()}")
        st.write("Valeurs manquantes par colonne :")
        st.dataframe(df_raw.isna().sum().to_frame("manquants"))

else:
    st.error("❌ Impossible de charger les données.")
