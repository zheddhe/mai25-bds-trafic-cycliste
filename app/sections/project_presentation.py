import streamlit as st
from pathlib import Path

URL_COMPTAGE_PARIS = "https://opendata.paris.fr/explore/dataset/\
comptage-velo-donnees-compteurs/information/?disjunctive.id_compteur&\
disjunctive.nom_compteur&disjunctive.id&disjunctive.name"

URL_COMPTEURS_PARIS = "https://parisdata.opendatasoft.com/explore/dataset/\
comptage-velo-compteurs/information/?disjunctive.counter&disjunctive.name&\
disjunctive.nom_compteur&disjunctive.id&disjunctive.id_compteur"

st.set_page_config(layout="wide")

st.title("⚙️ Projet Trafic Cycliste – Démarche & Résultats")

img = Path("app/assets/image_projet_mle.png")
if img.exists():
    st.image(str(img), use_container_width=True)
else:
    st.warning("Image not found: app/assets/image_projet_mle.png")

st.markdown("**Présentation synthétique (20 minutes)**")
with st.expander("🎙️ A - Introduction (1 min)", expanded=False):
    st.markdown("""
    La Ville de Paris dispose de compteurs permanents pour évaluer la
    pratique cycliste. Le but du projet : **prédire l’évolution horaire du
    trafic vélo** par site, pour :
    - Comprendre l'évolution du comptage par site et par heure.
    - Adapter les aménagements cyclables.
    - [*Bonus abandonné : évaluer l'influence du trafic Vélib*]

    👥 **Équipe** : Rémy Canal, Elias Djouadi, (Raphaël Parmentier)

    🎯 **Objectif** : mise en place d'une pipeline ML(Ops) complète +
    laboratoire interactif Streamlit
    """)

with st.expander("🔍 B - Exploration & Visualisation (8 min)", expanded=False):
    st.markdown("### 1. Données sources & nettoyages")
    st.markdown(f"""
    - **940k+ observations** sur **13 mois glissants** – source
    [Open Data Paris - Données Compteurs]({URL_COMPTAGE_PARIS})
    - Nettoyage des **doublons/valeurs manquantes** (clusters par nom de compteur)
    - **Reconstruction** des noms de compteur erronés
    """)

    with st.expander("Détails sur le Nettoyage", expanded=False):
        st.markdown(f"""
        Les valeurs manquantes sont regroupées sur des plages d'index contiguës,
        principalement pour des **colonnes liées aux photos et identifiants
        techniques**: `lien_vers_photo_du_site_de_comptage`,
        `identifiant_technique_compteur`, `id_photos`,
        `test_lien_vers_photos_du_site_de_comptage`, `id_photo_1`,
        `type_dimage`.

        D'autres clusters d'absence touchent aussi :
        `identifiant_du_compteur`, `identifiant_du_site_de_comptage`,
        `nom_du_site_de_comptage`, `date_d_installation_du_site_de_comptage`,
        `coordonnees_geographiques`, `url_sites`.

        Seuls quelques noms de compteur concentrent l'ensemble de ces
        observations manquantes (ex: '10 avenue de la Grande Armée...',
        '27 quai de la Tournelle...', etc.). Ces noms étaient des **valeurs
        erronées transitoires**; ils sont inférés et recoupés avec la base
         de données [Open Data Paris - Compteurs]({URL_COMPTEURS_PARIS}).

        Après reconduction des données d'origine, seules des données **techniques
        spécifiques de site** (URL/préfixe/suffixe/id) manquent encore. Les variables,
        associées, jugées peu explicatives, sont complètement écartées du dataset.
        """)

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            img = Path("app/assets/B/repartition_absence_de_valeur.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Répartition de l’absence de valeur (initiale)")
            else:
                st.warning("Image not found: app/assets/B/"
                           "repartition_absence_de_valeur.png")
        with col2:
            img = Path("app/assets/B/repartition_absence_de_valeur_corrige.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Répartition de l’absence de valeur (après nettoyage)")
            else:
                st.warning("Image not found: app/assets/B/"
                           "repartition_absence_de_valeur_corrige.png")

        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            st.markdown("""
            Ce graphique montre la répartition des valeurs manquantes par
            colonne **avant tout traitement**. On observe des **clusters
            d'absence** concentrés sur des colonnes techniques et liées aux photos.
            """)
        with col2:
            st.markdown("""
            Ce graphique montre les **variables restantes** après la phase
            de nettoyage et de correction.
            """)

    st.markdown("### 2. Feature engineering & enrichissement")
    with st.expander("Détails sur l'Enrichissement", expanded=False):
        st.markdown("#### Données intrinsèques / Recombinaisons")
        st.markdown("""
        Des variables complémentaires ont été créées à partir des données
        existantes pour la visualisation et la modélisation :
        - **`date_et_heure_de_comptage`** : extraction de l'année, mois
          (num/texte), jour (mois/année/semaine - num/texte), heure, semaine
          ISO-8601.
        - **`coordonnees_geographiques`** : extraction de la latitude et
          longitude.
        - **`nom_du_compteur`** : extraction de l'orientation du compteur.
        """)

        st.markdown("#### Données jours fériés")
        st.markdown("""
        Une information sur les jours fériés est récupérée et jointe via une
        API gouvernementale. Ces données sont connues 5 ans à l'avance et sont
        cruciales pour capter leur interaction avec le comportement des
        cyclistes.
        """)

        st.markdown("#### Données météo")
        st.markdown("""
        Des données météorologiques (température, code météo, précipitations,
        neige, altitude) sont intégrées depuis l'API Open-Meteo.com. Elles sont
        essentielles pour la modélisation du trafic à court terme.
        """)

        st.markdown("#### Données Vélib (envisagé mais non intégré)")
        st.markdown("""
        Nous avions envisagé d'utiliser des données des stations Vélib proches
        (nombre de stations ouvertes, bornes disponibles/occupées) via web
        scraping de l'API Vélib' Métropole pour des fins **explicatives
        uniquement**. Cette démarche n'a pas pu être intégrée en raison des
        contraintes de temps et de la nécessité d'un scraping sur une longue
        période.
        """)

    st.markdown("### 3. Visualisation et Statistiques")
    with st.expander("Analyse statistique univariée du comptage horaire",
                     expanded=False):
        col1, col2, col3 = st.columns([0.35, 0.35, 0.34])
        with col1:
            img = Path("app/assets/B/qq_plot_residus.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="QQ-plot des résidus du comptage horaire")
            else:
                st.warning("Image not found: app/assets/B/qq_plot_residus.png")
        with col2:
            img = Path("app/assets/B/analyse_comptage_horaire.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Analyse du comptage horaire (avant correction)")
            else:
                st.warning("Image not found: app/assets/B/analyse_comptage_horaire.png")
        with col3:
            img = Path("app/assets/B/nouvelle_distribution_comptage_horaire.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Analyse du comptage horaire global"
                         " (après correction)")
            else:
                st.warning("Image not found: app/assets/B/"
                           "nouvelle_distribution_comptage_horaire.png")
            img = Path("app/assets/B/nouvelle_repartition_comptage_horaire.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Analyse du comptage horaire par mois"
                         " (après correction)")
            else:
                st.warning("Image not found: app/assets/B/"
                           "nouvelle_repartition_comptage_horaire")

        col1, col2, col3 = st.columns([0.35, 0.35, 0.34])
        with col1:
            st.markdown("""
            Les données de la variable `comptage_horaire` **ne suivent pas une
            loi normale**, comme le montre l'écart entre les **quantiles
            théoriques** d’une loi normale (*courbe rouge*) et les valeurs
            observées (**en bleu**). Le **test d'Anderson** confirme cette
            non-normalité.
            """)
        with col2:
            st.markdown("""
            Ces graphiques illustrent la **distribution initiale** du comptage horaire
            (globale et par mois) avec une **concentration sur les faibles valeurs**
            et la présence d'**une valeur aberrante**.
            """)
        with col3:
            st.markdown("""
            Ce graphique montre la **distribution ajustée** du comptage horaire
            après la **correction de la valeur aberrante**, offrant une vue
            plus réaliste et confirmant la distribution sur les faibles valeurs.
            """)

    with st.expander("Data Visualisation du comptage horaire", expanded=False):
        col1, col2, col3 = st.columns([0.34, 0.34, 0.33])
        with col1:
            img = Path("app/assets/B/comptage_horaire_moyen_par_jour.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Comptage moyen par jour de la semaine")
            else:
                st.warning("Image not found: app/assets/B/"
                           "comptage_horaire_moyen_par_jour.png")
        with col3:
            img = Path("app/assets/B/Top_10_stations.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Top 10 des stations les plus fréquentées")
            else:
                st.warning("Image not found: app/assets/B/Top_10_stations.png")
        with col2:
            img = Path("app/assets/B/top_10_heures_fort_comptage.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Top 10 des heures avec le plus fort comptage total")
            else:
                st.warning("Image not found: app/assets/B/"
                           "top_10_heures_fort_comptage.png")

        col1, col2, col3 = st.columns([0.34, 0.34, 0.33])
        with col1:
            st.markdown("#### Comptage moyen par jour de la semaine")
            st.markdown("""
            Ce graphique met en lumière les différences d’intensité du trafic
            cycliste selon les jours de la semaine, en moyenne horaire.
            - **Jours les plus fréquentés** : Mardi (89.8) et Jeudi (89.6),
            reflétant une forte activité cycliste en milieu de semaine.
            Mercredi (87.9) est proche.
            - **Déclin en fin de semaine** : Le trafic baisse légèrement le
            Vendredi (≈81), puis significativement le Samedi (≈62) et surtout
            Dimanche (≈55), montrant une transition vers un usage loisir ou
            un abandon temporaire.
            - **Lundi plus faible** : Le Lundi (≈82) est en retrait,
            potentiellement dû à un démarrage de semaine plus progressif.
            """)
        with col2:
            st.markdown("#### Top 10 des heures avec le plus fort comptage total")
            st.markdown("""
            Ce graphique met en évidence les dix heures les plus fréquentées
            sur l’ensemble des stations.
            - **Heures de pointe très marquées** : La pointe du soir (**18h**)
            est en tête (plus de 7,5 millions), suivie de 19h, 17h et 20h.
            - **Heures de début de journée très actives** : Les créneaux 8h et
            9h enregistrent un fort comptage (> 5.9 millions), témoignant des
            déplacements domicile-travail.
            - **Heures de milieu de journée plus modérées** : 12h, 13h, 15h et
            16h affichent un comptage moindre (3.6 à 4.1 millions),
            correspondant à des déplacements personnels/professionnels.
            """)
        with col3:
            st.markdown("#### Top 10 des stations les plus fréquentées")
            st.markdown("""
            Ce graphique met en évidence une forte concentration du trafic
            cycliste sur quelques artères majeures de Paris.
            - **Domination du boulevard de Sébastopol et de la rue de Rivoli** :
            Le Totem 73 boulevard de Sébastopol (S-N) domine (plus de 3
            millions), avec deux stations de la rue de Rivoli également
            importantes. Ces itinéraires sont des corridors cyclables majeurs.
            - **Importance des compteurs bidirectionnels** : La présence de deux
            directions opposées sur plusieurs stations traduit des flux
            équilibrés.
            - **Diversité géographique** : D'autres stations comme Magenta,
            Ménilmontant, Voltaire et la Tournelle suggèrent une diffusion de
            l’usage au-delà de l’hyper-centre.
            """)

    with st.expander("Data Visualisation Comptage Vs météo et Vacances scolaires",
                     expanded=False):
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            img = Path("app/assets/B/boxplot_vacances_scolaires.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Boxplot Vacances Scolaires")
            else:
                st.warning("Image not found: app/assets/B/"
                           "boxplot_vacances_scolaires.png")
            st.markdown("""
            #### Comptage horaire Vs Vacances Scolaires
            TO COMPLETE
            """)
        with col2:
            img = Path("app/assets/B/boxplot_weather.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Boxplot Weather")
            else:
                st.warning("Image not found: app/assets/B/boxplot_weather.png")
            st.markdown("""
            #### Comptage horaire Vs Météo
            TO COMPLETE
            """)

    with st.expander("Data Visualisation Comptage Vs Géolocalisation",
                     expanded=False):
        col1, col2 = st.columns([0.5, 0.5])
        with col1:
            img = Path("app/assets/B/comptage_total_site.png")
            if img.exists():
                st.image(str(img), use_container_width=True,
                         caption="Comptage total par site (géolocalisation)")
            else:
                st.warning("Image not found: app/assets/B/comptage_total_site.png")
            st.markdown("""
            Une **disparité significative** du volume total de vélos
            enregistrés par les compteurs est observée entre le centre et la
            périphérie de Paris, ainsi qu'entre le nord-est et le sud-est.
            Cependant, la disparité du maillage des compteurs ne permet pas de
            tirer des conclusions certaines sur la seule base de la
            géolocalisation.
            """)
        with col2:
            boxplot_site_comptage = Path("app/assets/B/boxplot_site_comptage.png")
            if boxplot_site_comptage.exists():
                st.image(str(boxplot_site_comptage), use_container_width=True,
                         caption="Boxplot Site Comptage")
            else:
                st.warning("Image not found: app/assets/B/boxplot_site_comptage.png")
            st.markdown("""
            #### Comptage horaire par Site de comptage
            TO COMPLETE
            """)


with st.expander("🧪 C - Modélisation (10 à 12 min)", expanded=False):
    st.markdown("### 1. Objectif ML : Régression temporelle supervisée")

    with st.expander("Details Objectifs et Régression temporelle supervisée",
                     expanded=False):
        st.markdown("""
        - Variable cible : `comptage_horaire`
        - Extraction de variables explicatives additionnelles **spécifiques aux séries
        temporelles**
        - **AR** (*auto-régressives*) et **MM** (*moyennes mobiles*) depuis les valeurs
        précédentes de la **variables cible (prédite ou réelle)**
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
        - Modèles de **régression** étudiés : `linéaire`, `ElasticNet`, `KNN`,
        `RandomForest`, `XGBoost`, `SARIMAX`, `Deep Learning`
        """)
        with st.expander("Chronologie de l'étude des modèles", expanded=True):
            col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

            img = Path("app/assets/C/lineaire.png")
            if img.exists():
                col1.image(str(img), use_container_width=True)
            else:
                col1.warning("Image not found: app/assets/C/lineaire.png")

            img = Path("app/assets/C/elasticnet.png")
            if img.exists():
                col2.image(str(img), use_container_width=True)
            else:
                col2.warning("Image not found: app/assets/C/elasticnet.png")
            img = Path("app/assets/C/knn.png")

            if img.exists():
                col3.image(str(img), use_container_width=True)
            else:
                col3.warning("Image not found: app/assets/C/knn.png")

            img = Path("app/assets/C/random_forest.png")
            if img.exists():
                col4.image(str(img), use_container_width=True)
            else:
                col4.warning("Image not found: app/assets/C/random_forest.png")

            img = Path("app/assets/C/xgboost.png")
            if img.exists():
                col5.image(str(img), use_container_width=True)
            else:
                col5.warning("Image not found: app/assets/C/xgboost.png")

            img = Path("app/assets/C/sarimax.png")
            if img.exists():
                col6.image(str(img), use_container_width=True)
            else:
                col6.warning("Image not found: app/assets/C/sarimax.png")

            img = Path("app/assets/C/deep_learning.png")
            if img.exists():
                col7.image(str(img), use_container_width=True)
            else:
                col7.warning("Image not found: app/assets/C/deep_learning.png")

    st.markdown("### 2. Performances sans AR/MM")

    with st.expander("Details Performances sans AR/MM",
                     expanded=False):

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥇 **XGBoost** : meilleur compromis généralisation / précision
        """)
        img = Path("app/assets/C/xgboost_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/xgboost_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥈 **Random Forest** : excellent en entraînement mais problème de
        généralisation
        """)
        img = Path("app/assets/C/random_forest_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/random_forest_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥉 **KNN** : très bon en entraînement, mais problème de généralisation
        """)
        img = Path("app/assets/C/knn_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/knn_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **Régression linéaire** : mauvais en précision (stable en généralisation)
        """)
        img = Path("app/assets/C/lineaire_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/lineaire_mean_metrics.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **ElasticNet** : amélioration quasi inexistante de la régression linéaire
        malgré une grille de recherche des meilleurs hyperparamètres
        """)
        img = Path("app/assets/C/elasticnet_mean_metrics.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/elasticnet_mean_metrics.png")

    st.markdown("### 3. Performances explicatives avec AR/MM (avec valeurs réelles"
                " de la cible)")

    with st.expander("Details Performances explicatives avec AR/MM",
                     expanded=False):

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥇 **Random Forest** : parfait en entraînement et peu de problème de
        généralisation
        """)
        img = Path("app/assets/C/random_forest_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/"
                         "random_forest_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥈 **XGBoost** : excellent en entraînement et hyper stable en généralisation
        """)
        img = Path("app/assets/C/xgboost_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/xgboost_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥉 **KNN** : très bon en entraînement, mais problème de généralisation
        """)
        img = Path("app/assets/C/knn_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/knn_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🍫 **Régression linéaire** : honnête en précision (et stable en généralisation)
        """)
        img = Path("app/assets/C/lineaire_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/lineaire_mean_metrics_armm.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **ElasticNet** : amélioration inexistante de la régression linéaire
        malgré une grille de recherche des meilleurs hyperparamètres
        """)
        img = Path("app/assets/C/elasticnet_mean_metrics_armm.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/"
                         "elasticnet_mean_metrics_armm.png")

    st.markdown("### 4. Performances prédictives avec AR/MM en conditions réelles"
                " **step-by-step**")
    st.markdown("""
    Ce mode de prédiction sous entend qu'il n'y a **aucune utilisation des valeurs
    réelles** de la variable cible pour calculer les variables autorégressives ou
    moyennes mobiles, **seules les valeurs prédites** de la variable cible sont
    réutilisées récursivement pour ce calcul (*qui est du coup **bien plus couteux***)
    """)

    with st.expander("Details sur modèles standards",
                     expanded=False):

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥇 **Random Forest** : encore très bon en test avec une perte en généralisation
        (de ~8% en moyenne relativement au mode explicatif)
        """)
        img = Path("app/assets/C/random_forest_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/"
                         "random_forest_mean_metrics_forecast.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥉 **KNN** : une précision moindre mais une perte en généralisation
        la plus faible (de seulement ~6% en moyenne)
        """)
        img = Path("app/assets/C/knn_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/knn_mean_metrics_forecast.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - 🥈 **XGBoost** : une nette tendance à l'overfitting intuitivement liée
        à la nature même du modèle qui se base sur la correction itérative d'une erreur
        résiduelle
        """)
        img = Path("app/assets/C/xgboost_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/"
                         "xgboost_mean_metrics_forecast.png")

        col1, col2 = st.columns([1, 2])
        col1.markdown("""
        - ❌ **Régression linéaire** : médiocre en généralisation
        """)
        img = Path("app/assets/C/lineaire_mean_metrics_forecast.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/C/"
                         "lineaire_mean_metrics_forecast.png")

    st.markdown("### 5. Performances prédictives de modèles spécifiques avancés")
    st.markdown("""
    Les deux derniers modèles étudiés sont forcément sans vision des valeurs réelles
    de la variable cible et auto-calculent leur fenêtre de contexte
    - via la saison/AR/MA pour le modèle SARIMAX
    - via la fenêtre de contexte mobile pour les Transformers du modèle Deep Learning
    """)
    with st.expander("Details sur modèle SARIMAX",
                     expanded=False):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""
            - ❌ **SARIMAX** : très rapide en calcul de prédiction mais extrêmement
            couteux en entraînement et très faible en généralisation
            """)

    with st.expander("Details sur modèle Deep Learning",
                     expanded=False):
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            st.markdown("""
            - 🧠 **Deep Learning (Granite TTM)** :
            - excellent en *zero-shot*
            - améliorable par *fine-tuning* (R² ≈ 0.87 en test avec un contexte d'une
            semaine glissante: 168 lags)
            """)

with st.expander("🔚 D - Conclusion & Ouverture (1 min)", expanded=False):
    st.markdown("""
    ✅ Notre approche complète a exploré la **préparation** et l'**analyse statistique**
      des données de **séries temporelles**, leur **visualisation** et
      **modélisation ML** pour terminer sur un survol de l'état de l'art en
      **deep learning** sur ce sujet.

    🚀 **Prochaines étapes envisageables** :
    - Exploiter les possibilités des réseaux de neurones:
      - **profondeur** des données (8 ans d’archives disponible sur OpenData Paris)
      - **multiplication** des sources d'entraînement (via le transfert learning).
    - Exploiter la prédiction **multi-canal** de l'architecture Granite TTM
      (nécessite une **adaptation de la structure des données** - X colonnes
      `comptage_horaire` pour une même heure donnée pour capter les influences
      géographiques mutuelles)
    """)
    st.info("""
    👈 Les sections suivantes permettent d'explorer nos résultats intéractivement:
    - 🔍 Exploration statistique intéractive des données
    - 📈 Visualisation intéractive des données
    - 🧪 Laboratoire intéractif de modélisation
    """)

st.caption("Projet réalisé dans le cadre de la formation Machine Learning Engineering"
           " - DataScientest avril 2025")
