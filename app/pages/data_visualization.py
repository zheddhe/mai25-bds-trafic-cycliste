import streamlit as st
import pandas as pd
import plotly.express as px
from smartcheck.dataframe_common import load_dataset_from_config
import os


st.title("📈 Visualisations et Statistiques")


DATASET_NAME = "velo_comptage_refactored_data"
JOURS_ORDONNES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]


@st.cache_data
def cached_load_dataset_exploration():
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame({
            "nom_du_site_de_comptage": ["TEST_SITE"],
            "orientation_compteur": ["N-S"],
            "comptage_horaire": pd.Series([0], dtype="int"),
            "arrondissement": ["TEST_ARRONDISSEMENT"],
            "date_et_heure_de_comptage_hour": [1],
            'latitude': [48.8566],
            'longitude': [2.3522],
        })
    return load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)


with st.sidebar:
    if st.button("🔁 Recharger les données", key="reload_button"):
        cached_load_dataset_exploration.clear()  # type: ignore
        st.rerun()

df_raw = cached_load_dataset_exploration()

if df_raw is not None and isinstance(df_raw, pd.DataFrame):
    st.success("✅ Données chargées avec succès.")

    st.markdown("### 📦 Répartition des comptages horaires par mois")
    if "mois_annee_comptage" in df_raw.columns:
        fig_box_mois = px.box(
            df_raw,
            x="mois_annee_comptage",
            y="comptage_horaire",
            title="Répartition des comptages horaires par mois"
        )
        st.plotly_chart(fig_box_mois, use_container_width=True)

    st.markdown("### 📊 Comptage horaire moyen par jour de la semaine")
    if "date_et_heure_de_comptage_dayname" in df_raw.columns:
        df_jour = df_raw.groupby("date_et_heure_de_comptage_dayname")[
            "comptage_horaire"].mean().reset_index()
        df_jour["date_et_heure_de_comptage_dayname"] = pd.Categorical(
            df_jour["date_et_heure_de_comptage_dayname"],
            categories=JOURS_ORDONNES,
            ordered=True
        )
        df_jour = df_jour.sort_values("date_et_heure_de_comptage_dayname")

        top2_days = df_jour.nlargest(2, "comptage_horaire")[
            "date_et_heure_de_comptage_dayname"
        ].tolist()
        df_jour["couleur"] = df_jour[
            "date_et_heure_de_comptage_dayname"].apply(
            lambda d: "firebrick" if d in top2_days else "royalblue"
        )

        fig_jour = px.bar(
            df_jour,
            x="date_et_heure_de_comptage_dayname",
            y="comptage_horaire",
            color="couleur",
            color_discrete_map="identity",
            text_auto=True,
            title="Comptage horaire moyen par jour de la semaine",
            category_orders={
                "date_et_heure_de_comptage_dayname": JOURS_ORDONNES
            }
        )
        fig_jour.update_layout(showlegend=False)
        st.plotly_chart(fig_jour, use_container_width=True)

    st.markdown("### ⏱️ Top 10 des heures avec le plus fort comptage total")
    if "date_et_heure_de_comptage_hour" in df_raw.columns:
        df_heures = df_raw.groupby("date_et_heure_de_comptage_hour")[
            "comptage_horaire"].sum().reset_index()
        df_heures = df_heures.sort_values("comptage_horaire",
                                          ascending=False).head(10)

        top4_heures = df_heures.nlargest(4, "comptage_horaire")[
            "date_et_heure_de_comptage_hour"
        ].tolist()
        df_heures["couleur"] = df_heures[
            "date_et_heure_de_comptage_hour"].apply(
            lambda h: "firebrick" if h in top4_heures else "royalblue"
        )

        fig_heure = px.bar(
            df_heures,
            x="date_et_heure_de_comptage_hour",
            y="comptage_horaire",
            color="couleur",
            color_discrete_map="identity",
            text_auto=True,
            title="Top 10 des heures avec le plus fort comptage total"
        )
        fig_heure.update_layout(showlegend=False)
        st.plotly_chart(fig_heure, use_container_width=True)

    st.markdown("### 🚲 Top 10 des stations les plus fréquentées")
    if "nom_du_site_de_comptage" in df_raw.columns:
        df_stations = df_raw.groupby("nom_du_site_de_comptage")[
            "comptage_horaire"].sum().reset_index()
        df_stations = df_stations.sort_values(
            "comptage_horaire", ascending=False).head(10)
        fig_stations = px.bar(
            df_stations,
            x="comptage_horaire",
            y="nom_du_site_de_comptage",
            orientation='h',
            text_auto=True,
            color="comptage_horaire",
            color_continuous_scale="magma",
            title="Top 10 des stations les plus fréquentées"
        )
        fig_stations.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig_stations.update_layout(showlegend=False)
        st.plotly_chart(fig_stations, use_container_width=True)

    st.markdown("## 🌞 Distribution multi-niveaux (Sunburst)")
    with st.expander("🔧 Paramètres Sunburst"):
        col_path_1 = st.selectbox("Niveau 1", ["arrondissement",
                                               "nom_du_site_de_comptage",
                                               "date_et_heure_de_comptage_dayname"])
        col_path_2 = st.selectbox("Niveau 2", ["date_et_heure_de_comptage_hour",
                                               "date_et_heure_de_comptage_dayname"])
    if col_path_1 != col_path_2:
        fig_sun = px.sunburst(
            df_raw,
            path=[col_path_1, col_path_2],
            values="comptage_horaire",
            color=col_path_1,
            width=1000,
            height=800,
        )
        st.plotly_chart(fig_sun, use_container_width=True)
    else:
        st.warning("Les niveaux 1 et 2 doivent être différents pour Sunburst.")

    st.markdown("## 🗺️ Carte des sites de comptage")
    metric_col = "comptage_horaire"
    df_raw["arrondissement_num"] = df_raw[
        "arrondissement"
    ].str.extract(r"(\d+)").astype(float)

    df_grouped = df_raw.groupby(
        ['latitude', 'longitude', 'nom_du_site_de_comptage',
            'arrondissement', 'arrondissement_num'],
        as_index=False
    )[[metric_col]].sum()

    df_grouped = df_grouped.rename(columns={metric_col: 'comptage_total'})

    fig_map = px.scatter_map(
        df_grouped.sort_values("arrondissement_num"),
        lat='latitude',
        lon='longitude',
        size='comptage_total',
        color='arrondissement_num',
        hover_name='nom_du_site_de_comptage',
        hover_data=['arrondissement', 'comptage_total'],
        size_max=30,
        zoom=12,
        center={'lat': 48.8566, 'lon': 2.3522},
        title="Comptage total par site (Paris)",
        height=600,
        color_continuous_scale='Viridis'
    )
    fig_map.update_layout(
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        mapbox_style="carto-positron",
        coloraxis_colorbar=dict(
            title="Arrondissement",
            tickvals=list(range(1, 21)),
            ticktext=[f"{i}er" if i == 1 else f"{i}ème" for i in range(1, 21)]
        )
    )
    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.error("❌ Impossible de charger les données.")
