import streamlit as st
from pathlib import Path

URL_COMPTAGE_PARIS = "https://opendata.paris.fr/explore/dataset/\
comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&\
disjunctive.nom_compteur&disjunctive.id&disjunctive.name"
URL_COMPTEURS_PARIS = "https://parisdata.opendatasoft.com/explore/dataset/\
comptage-velo-compteurs/information/?disjunctive.counter&disjunctive.name&\
disjunctive.nom_compteur&disjunctive.id&disjunctive.id_compteur"
URL_API_METEO_PARIS = "https://open-meteo.com/en/docs/historical-forecast-api"
URL_API_JOUR_FERIES = "https://www.data.gouv.fr/fr/dataservices/jours-feries/"
URL_API_VACANCES = "https://data.education.gouv.fr/api"
URL_PAPIER_GRANITE = "https://arxiv.org/pdf/2401.03955"

st.set_page_config(layout="wide")

st.title("⚙️ Projet Trafic Cycliste – Démarche & Résultats")

img = Path("app/assets/image_projet_mle.png")
if img.exists():
    st.image(str(img), use_container_width=True)
else:
    st.warning("Image not found: app/assets/image_projet_mle.png")

st.markdown("**Présentation synthétique (20 minutes)**")
with st.expander("🎙️ A - Introduction (< 1 min)", expanded=False):
    st.markdown("""
    La Ville de Paris dispose de compteurs permanents pour évaluer la
    pratique cycliste.

    🎯 Le but du projet: **modéliser l’évolution horaire du trafic vélo** par site
    pour:
    - **Identifier les facteurs** qui influencent son évolution
    - Assister l'adaptation des aménagements cyclables en fonction des **prédictions
    de trafic**.

    🛠️ **Méthodologie**:
    - Mise en place d'un processus complet de modélisation basé sur l'apprentissage
    - Proposer un laboratoire d'expérimentation (modèles/données)

    👥 **Équipe**: Rémy Canal, Elias Djouadi, (Raphaël Parmentier)
    """)

with st.expander("🔍 B - Exploration & Visualisation (7 à 8 min)", expanded=False):
    st.markdown("### B.1. Données sources & nettoyages")
    st.markdown(f"""
    - **940k+ observations** sur **13 mois glissants** – source
    [Open Data Paris - Données Compteurs]({URL_COMPTAGE_PARIS})
    - Nettoyage des **doublons/valeurs manquantes** (clusters par nom de compteur)
    - **Reconstruction** des informations pour les noms de compteur erronés
    """)

    with st.expander("Détails sur le Nettoyage", expanded=False):
        col1, col2 = st.columns([0.5, 0.5])
        img = Path("app/assets/B/repartition_absence_de_valeur.png")
        if img.exists():
            col1.image(str(img), use_container_width=True)
        else:
            col1.warning("Image not found: app/assets/B/"
                         "repartition_absence_de_valeur.png")

        img = Path("app/assets/B/repartition_absence_de_valeur_corrige.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/B/"
                         "repartition_absence_de_valeur_corrige.png")

        col1, col2 = st.columns([0.5, 0.5])
        col1.markdown("""
        #### Cartographie des données manquantes (initiale)
        """)
        col2.markdown("""
        #### Cartographie des données manquantes (après nettoyage)
        """)

        st.markdown(f"""
        Les valeurs manquantes sont regroupées sur des **plages d'index contiguës**,
        principalement pour des **colonnes liées aux photos et identifiants
        techniques**: `lien_vers_photo_du_site_de_comptage`,
        `identifiant_technique_compteur`, `id_photos`,
        `test_lien_vers_photos_du_site_de_comptage`, `id_photo_1`,
        `type_dimage`.

        D'autres clusters d'absence touchent aussi:
        `identifiant_du_compteur`, `identifiant_du_site_de_comptage`,
        `nom_du_site_de_comptage`, `date_d_installation_du_site_de_comptage`,
        `coordonnees_geographiques`, `url_sites`.

        Seuls quelques noms de compteur concentrent l'ensemble de ces
        observations manquantes. Après analyse, il s'agit de noms avec des **valeurs
        erronées transitoires**; ils sont rectifiés et les données manquantes
        associées sont récupérées depuis la base de données
        [Open Data Paris - Compteurs]({URL_COMPTEURS_PARIS}).

        Au final, seules certaines des données **techniques spécifiques de site**
        (URL/préfixe/suffixe/id) manquent encore. Les variables, associées, jugées
        peu explicatives, sont complètement écartées du dataset.
        """)

    st.markdown("### B.2. Ingénierie sur les variables explicatives"
                " (Feature Engineering)")
    st.markdown("""
    - **Extraction** de données **intrinsèques** et **recombinaisons**
    - **Ajout** de données **gouvernementales**
    - **Croisement** avec données **météo**
    """)

    with st.expander("Détails sur l'Enrichissement", expanded=False):
        st.markdown("#### Données intrinsèques / Recombinaisons")
        st.markdown("""
        Des variables complémentaires ont été créées à partir des données
        existantes pour la visualisation et la modélisation:
        - **`date_et_heure_de_comptage`**: extraction de l'année, mois
          (num/texte), jour (mois/année/semaine - num/texte), heure, semaine
          ISO-8601.
        - **`coordonnees_geographiques`**: extraction de la latitude et
          longitude.
        - **`nom_du_compteur`**: extraction de l'orientation du compteur.
        """)

        st.markdown("#### Données jours fériés et vacances scolaires")
        st.markdown(f"""
        Les informations sur les jours fériés et vacances scolaires sont récupérées
        et jointe via les API gouvernementales [Jours Fériés]({URL_API_JOUR_FERIES})
        et [Vacances Scolaires]({URL_API_VACANCES}). Ces données sont connues plusieurs
        années à l'avance et sont intuitivement importantes pour expliquer l'évolution
        du trafic cycliste.
        """)

        st.markdown("#### Données météo")
        st.markdown(f"""
        Des données météorologiques (température, code météo, précipitations, neige,
        altitude) sont intégrées depuis l'API [Open-Meteo]({URL_API_METEO_PARIS}).
        Elles sont également pertinentes pour la modélisation du trafic à court terme.
        """)

        st.markdown("""
        >#### Données Vélib *(envisagées mais non intégrées au final)*
        >Nous avions envisagé d'utiliser des données des stations Vélib proches
        (nombre de stations ouvertes, bornes disponibles/occupées) via web
        scraping de l'API Vélib' Métropole pour des fins **explicatives
        uniquement**. Cette démarche n'a pas pu être intégrée en raison des
        contraintes de temps (nécessité d'un scraping sur une **longue
        période**).
        """)

    st.markdown("### B.3. Visualisation et Statistiques sur nos données")
    with st.expander("Analyses statistiques du **comptage horaire**",
                     expanded=False):
        col1, col2, col3 = st.columns([0.35, 0.35, 0.34])
        img = Path("app/assets/B/qq_plot_residus.png")
        if img.exists():
            col1.image(str(img), use_container_width=True)
        else:
            col1.warning("Image not found: app/assets/B/qq_plot_residus.png")

        img = Path("app/assets/B/analyse_comptage_horaire.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/B/analyse_comptage_horaire.png")

        img = Path("app/assets/B/analyse_comptage_horaire_corrige.png")
        if img.exists():
            col3.image(str(img), use_container_width=True)
        else:
            col3.warning("Image not found: app/assets/B/"
                         "analyse_comptage_horaire_corrige.png")

        col1, col2, col3 = st.columns([0.35, 0.35, 0.34])
        col1.markdown("""
        #### QQ-plot des données de comptage horaire
        Les données de la variable `comptage_horaire` **ne suivent pas une
        loi normale**, comme le montre l'écart entre les **quantiles
        théoriques** d’une loi normale (*courbe rouge*) et les valeurs
        observées (*en bleu*).

        Le **test d'Anderson** confirme cette
        non-normalité.
        """)

        col2.markdown("""
        #### Analyse du comptage horaire (avant correction)
        Ces graphiques illustrent la **distribution initiale** du `comptage_horaire`
        (globale et par mois) avec une **concentration sur les faibles valeurs**
        et la présence d'**une valeur aberrante**.
        """)

        col3.markdown("""
        #### Analyse du comptage horaire (après correction)
        Ces graphiques montrent la **distribution ajustée** du `comptage_horaire`
        après la **correction de la valeur aberrante**, offrant une vue
        plus réaliste et confirmant la distribution sur les faibles valeurs.
        """)

    with st.expander("Visualisation du **comptage horaire**", expanded=False):
        col1, col2, col3 = st.columns([0.34, 0.34, 0.33])
        img = Path("app/assets/B/comptage_horaire_moyen_par_jour_old.png")
        if img.exists():
            col1.image(str(img), use_container_width=True)
        else:
            col1.warning("Image not found: app/assets/B/"
                         "comptage_horaire_moyen_par_jour_old.png")

        img = Path("app/assets/B/top_10_heures_fort_comptage_old.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/B/"
                         "top_10_heures_fort_comptage_old.png")

        img = Path("app/assets/B/Top_10_stations_old.png")
        if img.exists():
            col3.image(str(img), use_container_width=True)
        else:
            col3.warning("Image not found: app/assets/B/Top_10_stations_old.png")

        col1, col2, col3 = st.columns([0.34, 0.34, 0.33])
        col1.markdown("#### Répartition du comptage moyen par jour")
        col1.markdown("""
        Ce graphique met en lumière les différences d’intensité du trafic
        cycliste selon les jours de la semaine, en moyenne horaire.
        - **Jours les plus fréquentés**: Mardi (89.8) et Jeudi (89.6),
        reflétant une forte activité cycliste en milieu de semaine.
        Mercredi (87.9) est proche.
        - **Déclin en fin de semaine**: Le trafic baisse légèrement le
        Vendredi (≈81), puis significativement le Samedi (≈62) et surtout
        Dimanche (≈55), montrant une transition vers un usage loisir ou
        un abandon temporaire.
        - **Lundi plus faible**: Le Lundi (≈82) est en retrait,
        potentiellement dû à un démarrage de semaine plus progressif.
        """)

        col2.markdown("#### Heures d'affluence pour le comptage total")
        col2.markdown("""
        Ce graphique met en évidence les dix heures les plus fréquentées
        sur l’ensemble des stations.
        - **Heures de pointe très marquées**: La pointe du soir (**18h**)
        est en tête (plus de 7,5 millions), suivie de 19h, 17h et 20h.
        - **Heures de début de journée très actives**: Les créneaux 8h et
        9h enregistrent un fort comptage (> 5.9 millions), témoignant des
        déplacements domicile-travail.
        - **Heures de milieu de journée plus modérées**: 12h, 13h, 15h et
        16h affichent un comptage moindre (3.6 à 4.1 millions),
        correspondant à des déplacements personnels/professionnels.
        """)

        col3.markdown("#### Stations les plus fréquentées")
        col3.markdown("""
        Ce graphique met en évidence une forte concentration du trafic
        cycliste sur quelques artères majeures de Paris.
        - **Domination du boulevard de Sébastopol et de la rue de Rivoli**:
        Le compteur Totem 73 boulevard de Sébastopol (S-N) domine (plus de 3
        millions), avec deux compteurs Rue de Rivoli également
        importants. Ces itinéraires sont des corridors cyclables majeurs.
        - **Importance des compteurs bidirectionnels**: La présence de deux
        directions opposées sur plusieurs stations traduit des flux
        équilibrés.
        - **Diversité géographique**: D'autres stations comme Magenta,
        Ménilmontant, Voltaire et la Tournelle suggèrent une diffusion de
        l’usage au-delà de l’hyper-centre.
        """)

    with st.expander("Visualisation du **comptage horaire** Vs **météo** et"
                     " **vacances scolaires**",
                     expanded=False):
        col1, col2 = st.columns([0.5, 0.5])
        img = Path("app/assets/B/boxplot_vacances_scolaires.png")
        if img.exists():
            col1.image(str(img), use_container_width=True)
        else:
            col1.warning("Image not found: app/assets/B/"
                         "boxplot_vacances_scolaires.png")
        col1.markdown("""
        #### Comptage horaire Vs Vacances Scolaires
        Ce graphique montre que le trafic cycliste est significativement
        plus **élevé** et plus **variable hors vacances scolaires**.

        Pendant les périodes de vacances, l'activité diminue,
        particulièrement pendant les **vacances de Noël** et le pont de
        **L'Ascension**.

        Cela indique que les vacances scolaires sont un facteur clé
        influençant l'usage du vélo.
        """)

        img = Path("app/assets/B/boxplot_weather.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/B/boxplot_weather.png")
        col2.markdown("""
        #### Comptage horaire Vs Météo
        Le graphique démontre une **forte corrélation** entre la météo
        et le trafic cycliste.

        L'affluence est plus **élevée** par **temps clément** ou **légèrement nuageux**,
        et **chute drastiquement** en cas de conditions sévères comme les **fortes
        pluies**, la **neige**, le **grésil**, le **verglas** ou les **orages**.

        La **variabilité** du trafic est également plus importante par
        **temps favorable**.
        """)

    with st.expander("Visualisation du **comptage horaire** Vs **géolocalisation**",
                     expanded=False):

        col1, col2 = st.columns([0.4, 0.6])
        img = Path("app/assets/B/comptage_total_site_old.png")
        if img.exists():
            col1.image(str(img), use_container_width=True)
        else:
            col1.warning("Image not found: app/assets/B/comptage_total_site.png")

        img = Path("app/assets/B/boxplot_site_comptage.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/B/boxplot_site_comptage.png")

        col1, col2 = st.columns([0.4, 0.6])
        col1.markdown("""
        #### Cartographie du comptage total par site
        Une **disparité significative** du volume total de vélos
        enregistrés par les compteurs est observée entre **le centre et la
        périphérie** de Paris, ainsi qu'entre **le nord-est et le sud-ouest**.

        Cependant, la **disparité** du maillage des compteurs ne permet pas de
        tirer des conclusions certaines sur la seule base de la
        géolocalisation.
        """)

        col2.markdown("""
        #### Distribution du comptage horaire par site
        On peut constater que le trafic est globalement **disparate** avec présence
        systématique d'outlier comme **schéma de répartition standard**
        - Les valeurs médianes sont **très faibles** mettant d'autant plus en évidence
        des **outliers nombreux**
        - Une station n'a que des **relevés nuls**: `108 avenue Denfert Rochereau`
        """)


with st.expander("🧪 C - Modélisation (10 à 11 min)", expanded=False):
    st.markdown("### C1. Objectif ML: Régression temporelle supervisée")

    with st.expander("Details Objectifs et Régression temporelle supervisée",
                     expanded=False):
        st.markdown("""
        - Variable cible: `comptage_horaire`
        - Métriques utilisées: `R²`, `RMSE`, `MAE` + observation des résidus dans
        le temps et de leur dérive
        - Extraction de variables explicatives additionnelles **spécifiques aux séries
        temporelles**
          - **AR** (*auto-régressives*) et **MM** (*moyennes mobiles*) depuis les
        valeurs précédentes de la **variables cible (prédite ou réelle)**
          - **Oscillatoires** (*de période P*) pour les composantes de l'horodatage
        suivant une période (heure -> 24, jour de la semaine -> 7, etc...)
        """)
        with st.expander("Formalisation mathématique", expanded=True):
            col1, col2 = st.columns([1.5, 2])
            col1.latex(r'''
                x_{AR_p} = y_{t - p}
            ''')
            col1.latex(r'''
                x_{MM_{sq}} = \frac{1}{s \times q} \sum_{i=1}^{s \times q + 1} y_{t - i}
            ''')

            col2.latex(r'''
                x_{\text{sinusoïdal}(x_p)} = \sin\left(2\pi \cdot \frac{x_p}{P}\right)
            ''')
            col2.latex(r'''
                x_{\text{cosinusoïdal}(x_p)} = \cos\left(2\pi \cdot \frac{x_p}{P}\right)
            ''')

        st.markdown("""
        - Modèles de **régression** étudiés: `linéaire`, `ElasticNet`, `KNN`,
        `RandomForest`, `XGBoost`, `SARIMAX`, `Deep Learning`
        """)
        with st.expander("Chronologie de l'étude des modèles", expanded=True):
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

            img = Path("app/assets/C/lin/lineaire.png")
            if img.exists():
                col1.image(str(img), use_container_width=True)
            else:
                col1.warning("Image not found: app/assets/C/lin/lineaire.png")

            img = Path("app/assets/C/elnet/elasticnet.png")
            if img.exists():
                col2.image(str(img), use_container_width=True)
            else:
                col2.warning("Image not found: app/assets/C/elnet/elasticnet.png")
            img = Path("app/assets/C/knn/knn.png")

            if img.exists():
                col3.image(str(img), use_container_width=True)
            else:
                col3.warning("Image not found: app/assets/C/knn/knn.png")

            img = Path("app/assets/C/rf/random_forest.png")
            if img.exists():
                col4.image(str(img), use_container_width=True)
            else:
                col4.warning("Image not found: app/assets/C/rf/random_forest.png")

            img = Path("app/assets/C/xgb/xgboost.png")
            if img.exists():
                col5.image(str(img), use_container_width=True)
            else:
                col5.warning("Image not found: app/assets/C/xgb/xgboost.png")

            img = Path("app/assets/C/sarx/sarimax.png")
            if img.exists():
                col6.image(str(img), use_container_width=True)
            else:
                col6.warning("Image not found: app/assets/C/sarx/sarimax.png")

            img = Path("app/assets/C/dl/deep_learning.png")
            if img.exists():
                col7.image(str(img), use_container_width=True)
            else:
                col7.warning("Image not found: app/assets/C/dl/deep_learning.png")

    st.markdown("### C.2. Performances explicatives sans AR/MM")

    with st.expander("Details Performances explivatives sans AR/MM", expanded=False):

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥇 **XGBoost**: meilleur compromis généralisation / précision
        """)
        img = Path("app/assets/C/xgb/xgboost_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/xgb/xgboost_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥈 **Random Forest**: excellent en entraînement mais problème de
        généralisation
        """)
        img = Path("app/assets/C/rf/random_forest_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/rf/"
                         "random_forest_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥉 **KNN**: très bon en entraînement, mais problème de généralisation
        """)
        img = Path("app/assets/C/knn/knn_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/knn/knn_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **Régression linéaire**: mauvais en précision (stable en généralisation)
        """)
        img = Path("app/assets/C/lin/lineaire_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/lin/lineaire_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **ElasticNet**: amélioration quasi inexistante de la régression linéaire
        malgré une grille de recherche des meilleurs hyperparamètres
        """)
        img = Path("app/assets/C/elnet/elasticnet_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/elnet/"
                         "elasticnet_mean_metrics.png")

        with st.expander("Meilleurs résultats:",
                         expanded=False):
            col1, col2, col3 = st.columns([1, 1, 1])
            img = Path("app/assets/C/xgb/xgboost_seb_sn_preds.png")
            if img.exists():
                col1.image(str(img), use_container_width=True)
            else:
                col1.warning("Image not found: app/assets/C/xgb/"
                             "xgboost_seb_sn_preds.png")

            img = Path("app/assets/C/xgb/xgboost_seb_sn_feats.png")
            if img.exists():
                col2.image(str(img), use_container_width=True)
            else:
                col2.warning("Image not found: app/assets/C/xgb/"
                             "xgboost_seb_sn_feats.png")

            img = Path("app/assets/C/xgb/xgboost_seb_sn_rot.png")
            if img.exists():
                col3.image(str(img), use_container_width=True)
            else:
                col3.warning("Image not found: app/assets/C/xgb/"
                             "xgboost_seb_sn_rot.png")
            # img = Path("app/assets/C/xgb/xgboost_seb_sn_trend.png")
            # if img.exists():
            #     col3.image(str(img), use_container_width=True)
            # else:
            #     col3.warning("Image not found: app/assets/C/xgb/"
            #                  "xgboost_seb_sn_trend.png")

    st.markdown("### C.3. Performances explicatives avec AR/MM (avec valeurs réelles"
                " de la cible)")

    with st.expander("Details Performances explicatives avec AR/MM", expanded=False):

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥇 **Random Forest**: parfait en entraînement et peu de problème de
        généralisation
        """)
        img = Path("app/assets/C/rf/random_forest_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/rf/"
                         "random_forest_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥈 **XGBoost**: excellent en entraînement et hyper stable en généralisation
        """)
        img = Path("app/assets/C/xgb/xgboost_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/xgb/"
                         "xgboost_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥉 **KNN**: très bon en entraînement, mais problème de généralisation
        """)
        img = Path("app/assets/C/knn/knn_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/knn/knn_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🍫 **Régression linéaire**: honnête en précision (et stable en généralisation)
        """)
        img = Path("app/assets/C/lin/lineaire_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/lin/"
                         "lineaire_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **ElasticNet**: amélioration inexistante de la régression linéaire
        malgré une grille de recherche des meilleurs hyperparamètres
        """)
        img = Path("app/assets/C/elnet/elasticnet_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/elnet"
                         "elasticnet_mean_metrics_armm.png")

        with st.expander("Meilleurs résultats:",
                         expanded=False):
            col1, col2, col3 = st.columns([1, 1, 1])
            img = Path("app/assets/C/rf/random_forest_seb_sn_preds_armm.png")
            if img.exists():
                col1.image(str(img), use_container_width=True)
            else:
                col1.warning("Image not found: app/assets/C/rf/"
                             "random_forest_seb_sn_preds_armm.png")

            img = Path("app/assets/C/rf/random_forest_seb_sn_feats_armm.png")
            if img.exists():
                col2.image(str(img), use_container_width=True)
            else:
                col2.warning("Image not found: app/assets/C/rf/"
                             "random_forest_seb_sn_feats_armm.png")

            img = Path("app/assets/C/rf/random_forest_seb_sn_rot_armm.png")
            if img.exists():
                col3.image(str(img), use_container_width=True)
            else:
                col3.warning("Image not found: app/assets/C/rf/"
                             "random_forest_seb_sn_rot_armm.png")
            # img = Path("app/assets/C/rf/random_forest_seb_sn_trend_armm.png")
            # if img.exists():
            #     col3.image(str(img), use_container_width=True)
            # else:
            #     col3.warning("Image not found: app/assets/C/rf/"
            #                  "random_forest_seb_sn_trend_armm.png")

    st.markdown("### C.4. Performances prédictives avec AR/MM en conditions réelles"
                " **step-by-step**")
    st.markdown("""
    Ce mode de prédiction sous entend qu'il n'y a **aucune utilisation des valeurs
    réelles** de la variable cible pour calculer les variables autorégressives ou
    moyennes mobiles, **seules les valeurs prédites** de la variable cible sont
    réutilisées récursivement pour ce calcul (*qui est du coup **bien plus couteux***)
    """)

    with st.expander("Details Performance en conditions réelles sur modèles standards",
                     expanded=False):

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥇 **Random Forest**: encore très bon en test avec une perte en généralisation
        (de ~8% en moyenne relativement au mode explicatif)
        """)
        img = Path("app/assets/C/rf/random_forest_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/rf/"
                         "random_forest_mean_metrics_forecast.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥈 **KNN**: une précision moindre mais une perte en généralisation
        la plus faible (de seulement ~6% en moyenne)
        """)
        img = Path("app/assets/C/knn/knn_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/knn/"
                         "knn_mean_metrics_forecast.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥉 **XGBoost**: une nette tendance à l'overfitting intuitivement liée
        à la nature même du modèle qui se base sur la correction itérative d'une erreur
        résiduelle
        """)
        img = Path("app/assets/C/xgb/xgboost_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/xgb/"
                         "xgboost_mean_metrics_forecast.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **Régression linéaire**: médiocre en généralisation
        """)
        img = Path("app/assets/C/lin/lineaire_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/lin"
                         "lineaire_mean_metrics_forecast.png")

        with st.expander("Meilleurs résultats:",
                         expanded=False):
            col1, col3 = st.columns([1, 1])
            img = Path("app/assets/C/rf/random_forest_seb_sn_preds_forecast.png")
            if img.exists():
                col1.image(str(img), use_container_width=True)
            else:
                col1.warning("Image not found: app/assets/C/rf/"
                             "random_forest_seb_sn_preds_forecast.png")

            # img = Path("app/assets/C/rf/random_forest_seb_sn_wide_preds_forecast.png")
            # if img.exists():
            #     col2.image(str(img), use_container_width=True)
            # else:
            #     col2.warning("Image not found: app/assets/C/rf/"
            #                  "random_forest_seb_sn_wide_preds_forecast.png")

            img = Path("app/assets/C/rf/random_forest_seb_sn_rot_forecast.png")
            if img.exists():
                col3.image(str(img), use_container_width=True)
            else:
                col3.warning("Image not found: app/assets/C/rf/"
                             "random_forest_seb_sn_rot_forecast.png")
            # img = Path("app/assets/C/rf/random_forest_seb_sn_trend_forecast.png")
            # if img.exists():
            #     col3.image(str(img), use_container_width=True)
            # else:
            #     col3.warning("Image not found: app/assets/C/rf/"
            #                  "random_forest_seb_sn_trend_forecast.png")

    st.markdown("### C.5. Performances prédictives de modèles spécifiques avancés")
    st.markdown(f"""
    Les deux derniers modèles étudiés utilisent par contruction uniquement les valeurs
    prédites calculées de la variable cible et auto-calculent leur contexte mobile
    - via la **saisonnalité/AR/MA** pour le modèle **SARIMAX**
    - via la **fenêtre de contexte mobile** (FCM) pour les Tiny Time Mixers (TTM)
    du modèle de **Deep Learning** [Granite d'IBM]({URL_PAPIER_GRANITE})
    """)
    with st.expander("Details Performance en conditions réelles sur modèle SARIMAX",
                     expanded=False):
        col1, col2, col3 = st.columns([1, 1, 1])
        col1.markdown("""
        - ❌ **SARIMAX**:
          - **Avantage:**
            - Très rapide en calcul de prédiction (le plus rapide
            de tous nos modèles)
          - **Inconvénients**:
            - **Très faible en généralisation** notamment sur les évolutions atypiques
              ##### **R² train = 0.907 | R² test = 0.645**
              ➡️ obtenus avec notre meilleure combinaison de saison et d'ordre
              raisonnablement **calculable**
            - **Extrêmement couteux** en temps de calcul en entraînement
              - **impossible** de mettre en oeuvre les ordres élevés détectés suite à
              l'analyse ACF (corrélation complète) et PACF (corrélation partielle)
              - **impossible** également d'explorer des saisons **plus longues**
              (ou comme on le suppose également des **saisonnalités multiples**)
              - **difficile** de traiter les données complètes d'un compteur
              à degré élevé (car **complexité en O(n.k²)**)
        """)

        img = Path("app/assets/C/sarx/sarimax_seasonal_decompose.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/sarx/"
                         "sarimax_seasonal_decompose.png")

        img = Path("app/assets/C/sarx/sarimax_seb_sn_acf_pacf.png")
        if img.exists():
            col3.image(str(img), use_container_width=True)
        else:
            col3.warning("Image not found: app/assets/C/sarx/"
                         "sarimax_seb_sn_acf_pacf.png")

        with st.expander("Meilleurs résultats: **(p,d,q)=(3,1,3)"
                         " (P,D,Q,S)=(3,1,3,24)**",
                         expanded=False):
            col1, col3 = st.columns([1, 1])
            img = Path("app/assets/C/sarx/sarimax_seb_sn_preds_313-313-24.png")
            if img.exists():
                col1.image(str(img), use_container_width=True)
            else:
                col1.warning("Image not found: app/assets/C/sarx/"
                             "sarimax_seb_sn_preds_313-313-24.png")

            # img = Path("app/assets/C/sarx/sarimax_seb_sn_wide_preds_313-313-24.png")
            # if img.exists():
            #     col2.image(str(img), use_container_width=True)
            # else:
            #     col2.warning("Image not found: app/assets/C/sarx/"
            #                  "sarimax_seb_sn_wide_preds_313-313-24.png")

            img = Path("app/assets/C/sarx/sarimax_seb_sn_rot_313-313-24.png")
            if img.exists():
                col3.image(str(img), use_container_width=True)
            else:
                col3.warning("Image not found: app/assets/C/sarx/"
                             "sarimax_seb_sn_rot_313-313-24.png")
            # img = Path("app/assets/C/sarx/sarimax_seb_sn_trend_313-313-24.png")
            # if img.exists():
            #     col3.image(str(img), use_container_width=True)
            # else:
            #     col3.warning("Image not found: app/assets/C/sarx/"
            #                  "sarimax_seb_sn_trend_313-313-24.png")

    with st.expander("Details Performance en conditions réelles sur modèle"
                     " Deep Learning",
                     expanded=False):
        col1, col2, col3 = st.columns([1, 1, 1])
        col1.markdown("""
        - 🧠 **Deep Learning (Granite TTM)**:
          - Déjà très bon et généralisable en *zero-shot*:
            ##### **R² train = 0.873 | R² test = 0.891**
            ➡️ obtenu sans **aucune données exogène**
          - Améliorable en *fine-tuning* aux possibilités théoriques **très
        nombreuses** (**transfert learning** entre compteurs, **prédiction parallèle
        (multi-canal)**, **pas** de prédiction, **mixage des variables**...):
            ##### **R² train = 0.855 | R² test = 0.871**
            ➡️ obtenu avec une FCM de taille 168 (= 1 semaine soit **29M de poids**
            à affiner) et entrainé sur 60% de **tous les compteurs
            disponibles**
        """)

        img = Path("app/assets/C/dl/deep_learning_granite_ttm_arch.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/dl/"
                         "deep_learning_granite_ttm_arch.png")

        img = Path("app/assets/C/dl/deep_learning_int_preds.png")
        if img.exists():
            col3.image(str(img), use_container_width=True)
        else:
            col3.warning("Image not found: app/assets/C/dl/"
                         "deep_learning_int_preds.png")

        with st.expander("Meilleurs résultats: **zeroshot**", expanded=False):
            col1, col3 = st.columns([1, 1])
            img = Path("app/assets/C/dl/deep_learning_preds.png")
            if img.exists():
                col1.image(str(img), use_container_width=True)
            else:
                col1.warning("Image not found: app/assets/C/dl/"
                             "deep_learning_preds.png")

            # img = Path("app/assets/C/dl/deep_learning_wides_preds.png")
            # if img.exists():
            #     col2.image(str(img), use_container_width=True)
            # else:
            #     col2.warning("Image not found: app/assets/C/dl/"
            #                  "deep_learning_wides_preds.png")

            img = Path("app/assets/C/dl/deep_learning_rot.png")
            if img.exists():
                col3.image(str(img), use_container_width=True)
            else:
                col3.warning("Image not found: app/assets/C/dl/"
                             "deep_learning_rot.png")
            # img = Path("app/assets/C/dl/deep_learning_trend.png")
            # if img.exists():
            #     col3.image(str(img), use_container_width=True)
            # else:
            #     col3.warning("Image not found: app/assets/C/dl/"
            #                  "deep_learning_trend.png")

with st.expander("🔚 D - Conclusion & Ouverture (< 1 min)", expanded=False):
    st.markdown("""
    ✅ Notre approche complète a exploré la **préparation** et l'**analyse statistique**
      des données de **séries temporelles**, leur **visualisation** et
      **modélisation ML** pour terminer sur un survol de l'état de l'art en
      **deep learning** sur ce sujet.

    🚀 **Prochaines étapes envisageables**:
    - Exploiter les possibilités des réseaux de neurones:
      - **profondeur** des données (8 ans d’archives disponible sur OpenData Paris)
      - **multiplication** des sources d'entraînement (via le transfert learning).
    - Exploiter la prédiction **multi-canal** de l'architecture Granite TTM
      pour capter les influences géographiques mutuelles.
    """)
    st.info("""
    👈 Les sections suivantes permettent d'explorer nos résultats intéractivement:
    - 🔍 Exploration statistique intéractive des données
    - 📈 Visualisation intéractive des données
    - 🧪 Laboratoire intéractif de modélisation
    """)

st.caption("Projet réalisé dans le cadre de la formation Machine Learning Engineering"
           " - DataScientest avril 2025")
