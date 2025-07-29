import streamlit as st
from smartcheck.dataframe_common import load_dataset_from_config
from smartcheck.dataframe_project_specific import get_missing_periods
import pandas as pd
import os
import plotly.express as px
import plotly.graph_objects as go


# --- Chargement des données ---
@st.cache_data
def cached_get_missing_periods(df_raw):
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame({
            "start": [pd.to_datetime("2025-01-07T11:00:00+01:00")],
            "end": [pd.to_datetime("2025-01-07T14:00:00+01:00")],
            "site": ["TEST_SITE"],
            "direction": ["N-S"],
            "label": ["TEST_SITE - N-S"],
        })
    df = get_missing_periods(df_raw)
    return df


@st.cache_data
def cached_load_dataset_exploration():
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame({
            "nom_du_site_de_comptage": ["TEST_SITE", "TEST_SITE"],
            "orientation_compteur": ["N-S", "N-S"],
            "comptage_horaire": [0, 0],
            "date_et_heure_de_comptage": [
                "2025-01-07T11:00:00+01:00",
                "2025-01-07T14:00:00+01:00",
            ],
        })
    df = load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)
    return df


# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_refactored_data"

st.title("🔍 Exploration statistique intéractive des données")
st.markdown("""
Cette page vous permet d'explorer les données afin d'observer
différents types de statistiques mesurant la qualité et la pertinence des données de
comptage vélo avec des paramètres de regroupement personnalisables.
""")
st.info("""
👈 Le dataset est préchargé mais vous pouvez forcer son rechargement depuis google
drive via le menu
""")

with st.sidebar:
    if st.button("🔁 Rechargement du Dataset", key="reload_button"):
        cached_load_dataset_exploration.clear()  # type: ignore
        st.rerun()

with st.spinner("⏳ Chargement du dataset en cours..."):
    df_raw = cached_load_dataset_exploration()

if df_raw is not None and isinstance(df_raw, pd.DataFrame):
    st.success(f"✅ Données [{DATASET_NAME}] chargées avec succès.")

    with st.expander("🚨 Identification des doublons et des discontinuités"
                     "dans les périodes de relevé horaire"):
        obs_dup = df_raw.duplicated(
            subset=["nom_du_site_de_comptage",
                    "orientation_compteur",
                    "date_et_heure_de_comptage"],
            keep=False,
        )
        st.markdown("""
        **Observations dupliquées** (même compteur et même heure présents plusieurs
        fois):
        """)
        st.dataframe(df_raw[obs_dup])
        missing_df = cached_get_missing_periods(df_raw)
        fig = px.timeline(
            missing_df,
            x_start="start",
            x_end="end",
            y="label",
            # color="label",
            title="Compteurs avec périodes manquantes"
        )
        fig = go.Figure(fig)
        fig.update_yaxes(autorange="reversed")
        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor="lightgray",
            tickfont=dict(size=10)
        )
        fig.update_layout(showlegend=False, height=800)
        st.plotly_chart(fig, use_container_width=True)

    with st.expander("📊 Statistiques descriptives globales"):
        st.dataframe(df_raw.describe(include='all').T)

    with st.expander("🧮 Statistiques simples après regroupements autour de "
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

else:
    st.error("❌ Impossible de charger les données.")
