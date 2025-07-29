import streamlit as st
import pandas as pd
import plotly.express as px
from smartcheck.dataframe_common import load_dataset_from_config
import os

# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_refactored_data"
JOURS_ORDONNES = ["Monday", "Tuesday", "Wednesday", "Thursday",
                  "Friday", "Saturday", "Sunday"]
MOIS_ORDONNES = ["January", "February", "March", "April",
                 "May", "June", "July", "August",
                 "September", "October", "November", "December"]
HEURES_ORDONNEES = ["0h", "1h", "2h", "3h", "4h", "5h", "6h", "7h", "8h", "9h", "10h",
                    "11h", "12h", "13h", "14h", "15h", "16h", "17h", "18h", "19h",
                    "20h", "21h", "22h", "23h",]


@st.cache_data
def cached_load_dataset_visualization():
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame({
            "nom_du_site_de_comptage": ["TEST_SITE"],
            "orientation_compteur": ["N-S"],
            "comptage_horaire": pd.Series([0], dtype="int"),
            "arrondissement": ["TEST_ARRONDISSEMENT"],
            "date_et_heure_de_comptage_hour": [1],
            "date_et_heure_de_comptage_dayname": ["Monday"],
            "date_et_heure_de_comptage_monthname": ["January"],
            'latitude': [48.8566],
            'longitude': [2.3522],
        })
    return load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)


st.title("📈 Visualisations intéractives des données")
st.markdown("""
Cette page vous permet de plonger dans différentes visualisation ces données de
comptage vélo avec des paramètres personnalisables et des graphiques interactifs.
""")
st.info("""
👈 Le dataset est préchargé mais vous pouvez forcer son rechargement depuis google
drive via le menu
""")

with st.sidebar:
    if st.button("🔁 Rechargement du Dataset", key="reload_button"):
        cached_load_dataset_visualization.clear()  # type: ignore
        st.rerun()

with st.spinner("⏳ Chargement du dataset en cours..."):
    df_raw = cached_load_dataset_visualization()

if df_raw is not None and isinstance(df_raw, pd.DataFrame):
    st.success(f"✅ Données [{DATASET_NAME}] chargées avec succès.")

    st.markdown("")
    column_title_distrib, column_period_distrib = st.columns([2, 1])
    column_title_distrib.markdown("### 📈 Distribution du comptage horaire")
    period_distrib = column_period_distrib.selectbox(
        "Périodicité",
        ["date_et_heure_de_comptage_monthname",
         "date_et_heure_de_comptage_dayname",
         "date_et_heure_de_comptage_hour"]
    )
    df_period = df_raw.copy()
    if period_distrib == "date_et_heure_de_comptage_dayname":
        categories_distrib = JOURS_ORDONNES
    elif period_distrib == "date_et_heure_de_comptage_monthname":
        categories_distrib = MOIS_ORDONNES
    else:
        categories_distrib = HEURES_ORDONNEES
        df_period[period_distrib] = df_period[period_distrib].apply(lambda x: f"{x}h")
    df_period[period_distrib] = pd.Categorical(
        df_period[period_distrib],
        categories=categories_distrib,
        ordered=True
    )
    if period_distrib in df_raw.columns:
        fig_box_mois = px.box(
            df_period,
            x=period_distrib,
            y="comptage_horaire",
            points=False,  # delete individual points
            category_orders={
                period_distrib: categories_distrib
            }
        )
        fig_box_mois.update_traces(
            hovertemplate=(
                "Q1 = %{q1}<br>" +
                "Median = %{median}<br>" +
                "Q3 = %{q3}<br>" +
                "Lower Fence = %{lowerfence}<br>" +
                "Upper Fence = %{upperfence}<br><extra></extra>"
            )
        )
        st.plotly_chart(fig_box_mois, use_container_width=True)

    column_title_mean, column_period_mean = st.columns([2, 1])
    column_title_mean.markdown("### 📊 Comptage horaire moyen")
    period_mean = column_period_mean.selectbox(
        "Périodicité",
        ["date_et_heure_de_comptage_dayname",
         "date_et_heure_de_comptage_monthname"]
    )
    if period_mean == "date_et_heure_de_comptage_dayname":
        categories_mean = JOURS_ORDONNES
    else:
        categories_mean = MOIS_ORDONNES
    if period_mean in df_raw.columns:
        df_period = df_raw.groupby(period_mean)[
            "comptage_horaire"].mean().reset_index()
        df_period[period_mean] = pd.Categorical(
            df_period[period_mean],
            categories=categories_mean,
            ordered=True
        )
        df_period = df_period.sort_values(period_mean)

        top2_period = df_period.nlargest(2, "comptage_horaire")[period_mean].tolist()
        df_period["couleur"] = df_period[
            period_mean].apply(
            lambda d: "firebrick" if d in top2_period else "royalblue"
        )

        fig_jour = px.bar(
            df_period,
            x=period_mean,
            y="comptage_horaire",
            color="couleur",
            color_discrete_map="identity",
            text_auto=True,
            category_orders={
                period_mean: categories_mean
            }
        )
        fig_jour.update_layout(showlegend=False)
        st.plotly_chart(fig_jour, use_container_width=True)

    column_title_count, column_period_count = st.columns([2, 1])
    column_title_count.markdown("### 🧮 Comptage horaire total")
    period_count = column_period_count.selectbox(
        "Périodicité",
        ["date_et_heure_de_comptage_hour",
         "date_et_heure_de_comptage_dayname",]
    )
    df_period = df_raw.copy()
    if period_count == "date_et_heure_de_comptage_dayname":
        categories_count = JOURS_ORDONNES
    else:
        categories_count = HEURES_ORDONNEES
        df_period[period_count] = df_period[period_count].apply(lambda x: f"{x}h")
    if period_count in df_raw.columns:
        df_period = df_period.groupby(period_count)[
            "comptage_horaire"].sum().reset_index()
        df_period = df_period.sort_values("comptage_horaire",
                                          ascending=False)

        top2_period = df_period.nlargest(2, "comptage_horaire")[
            period_count
        ].tolist()
        df_period["couleur"] = df_period[
            period_count].apply(
            lambda h: "firebrick" if h in top2_period else "royalblue"
        )

        fig_heure = px.bar(
            df_period,
            x=period_count,
            y="comptage_horaire",
            color="couleur",
            color_discrete_map="identity",
            text_auto=True,
            category_orders={
                period_count: categories_count
            }
        )
        fig_heure.update_layout(showlegend=False)
        st.plotly_chart(fig_heure, use_container_width=True)

    column_title_sites, column_top_sites = st.columns([2, 1])
    column_title_sites.markdown("### 🚲 Sites avec le plus de passage")
    top_sites = column_top_sites.number_input(
        "Top à afficher",
        min_value=1,
        max_value=67,
        value=10,
        key="top_sites_inp"
    )
    if "nom_du_site_de_comptage" in df_raw.columns:
        df_stations = df_raw.groupby("nom_du_site_de_comptage")[
            "comptage_horaire"].sum().reset_index()
        df_stations = df_stations.sort_values(
            "comptage_horaire", ascending=False).head(top_sites)
        fig_stations = px.bar(
            df_stations,
            x="comptage_horaire",
            y="nom_du_site_de_comptage",
            orientation='h',
            text_auto=True,
            color="comptage_horaire",
            color_continuous_scale="viridis",
        )
        fig_stations.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig_stations.update_layout(showlegend=False)
        st.plotly_chart(fig_stations, use_container_width=True)

    column_title_multi, column_path_1, column_path_2 = st.columns([3, 1, 1])
    column_title_multi.markdown("## 🗂️ Distribution multi-niveaux du comptage")
    col_path_1 = column_path_1.selectbox(
        "Niveau 1",
        ["arrondissement",
            "nom_du_site_de_comptage",
            "date_et_heure_de_comptage_dayname"],
        key="col_path_1_sb"
    )
    col_path_2 = column_path_2.selectbox(
        "Niveau 2",
        ["date_et_heure_de_comptage_hour",
            "date_et_heure_de_comptage_dayname"],
        key="col_path_2_sb"
    )
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
