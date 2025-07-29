import streamlit as st
from pathlib import Path

st.set_page_config(layout="wide")

st.title("⚙️ Projet Trafic Cycliste – Démarche & Résultats")

img = Path("app/assets/image_projet_mle.png")
if img.exists():
    st.image(str(img), use_container_width=True)
else:
    st.warning("Image not found: app/assets/image_projet_mle.png")

st.markdown("**Présentation synthétique (20 minutes)**")
with st.expander("🎙️ A - Introduction (1 min)", expanded=False):
    st.markdown(
        """
    La Ville de Paris dispose de compteurs permanents pour évaluer la
    pratique cycliste. Le but du projet : **prédire l’évolution horaire du
    trafic vélo** par site, pour :
    - Adapter les aménagements cyclables
    - Détecter les zones à surveiller
    - Évaluer l’impact des facteurs météo, jour férié, horaire, etc.

    👥 **Équipe** : Rémy Canal, Elias Djouadi, (Raphaël Parmentier)

    🧠 **Méthodologie** : mise en place d'une pipeline ML(Ops) complète +
    laboratoire interactif Streamlit
    """
    )

with st.expander("🔍 B - Exploration & Visualisation (8 min)", expanded=False):
    st.markdown("### 1. Données sources & nettoyages")
    st.markdown(
        """
    - 940k+ observations, 13 mois glissants – source OpenData Paris
    - Nettoyage des doublons/valeurs manquantes (clusters par nom de compteur)
    - **Reconstruction automatique** de noms de compteur erronés
    """
    )

    with st.expander("Détails sur le Nettoyage", expanded=False):
        st.markdown("#### 1.1 Traitement des données manquantes")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            **Répartition de l'absence de valeur (initiale)**
            Ce graphique montre la répartition des valeurs manquantes par
            colonne avant tout traitement. On observe des clusters d'absence
            concentrés sur des colonnes techniques et liées aux photos.
            """
            )
        with col2:
            graphique_repartition_de_valeur_manquantes = Path(
                "app/assets/répartition de l’absence de valeur.png"
            )
            if graphique_repartition_de_valeur_manquantes.exists():
                st.image(
                    str(graphique_repartition_de_valeur_manquantes),
                    use_container_width=True,
                    caption="Répartition de l’absence de valeur (initiale)"
                )
            else:
                st.warning(
                    "Image not found: app/assets/répartition de l’absence"
                    " de valeur.png"
                )

        st.markdown(
            """
        Les valeurs manquantes sont regroupées sur des plages d'index contiguës
        ,principalement pour des colonnes liées aux photos et identifiants
        techniques : `lien_vers_photo_du_site_de_comptage`,
        `identifiant_technique_compteur`, `id_photos`,
        `test_lien_vers_photos_du_site_de_comptage`, `id_photo_1`,
        `type_dimage`.

        D'autres clusters d'absence touchent aussi :
        `identifiant_du_compteur`, `identifiant_du_site_de_comptage`,
        `nom_du_site_de_comptage`, `date_d_installation_du_site_de_comptage`,
        `coordonnees_geographiques`, `url_sites`.

        Seuls quelques noms de compteur concentrent l'ensemble de ces
        observations manquantes (ex: '10 avenue de la Grande Armée...',
        '27 quai de la Tournelle...', etc.). Ces noms étaient des valeurs
        erronées transitoires ; ils ont été inférés et recoupés avec la base
        `Comptage vélo - Compteurs`.

        Après reconduction des données d'origine, seules les données techniques
        spécifiques de site (URL/préfixe/suffixe/id) subsistent. Ces variables,
        jugées peu explicatives, ont été abandonnées.
        """
        )

        st.markdown("#### 1.2 Traitement des types")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            **Répartition de l'absence de valeur (corrigée)**
            Ce graphique montre l'état des valeurs manquantes après la phase
            de nettoyage et de correction. La majorité des informations
            techniques non essentielles ont été traitées, laissant un
            dataset plus propre.
            """
            )
        with col2:
            graphique_repartition_de_valeur_manquantes_corrige = Path(
                "app/assets/répartition de l’absence de valeur corrigé.png"
            )
            if graphique_repartition_de_valeur_manquantes_corrige.exists():
                st.image(
                    str(graphique_repartition_de_valeur_manquantes_corrige),
                    use_container_width=True,
                    caption="Répartition de l’absence de valeur (corrigée)"
                )
            else:
                st.warning(
                    "Image not found: app/assets/répartition de l’absence"
                    " de valeur corrigé.png"
                )

        st.markdown(
            """
        Nous avons ajusté le type de certaines variables et extrait des
        informations périodiques (année, mois, jour, heure, semaine) de la
        variable `date_et_heure_de_comptage`. Des informations géographiques
        (latitude,longitude) ont été extraites de `coordonnees_geographiques`
        et l'orientation du compteur du `nom_du_compteur`.
        """
        )

    st.markdown("### 2. Feature engineering & enrichissement")
    with st.expander("Détails sur l'Enrichissement", expanded=False):
        st.markdown("#### 2.1 Données intrinsèques / Recombinaisons")
        st.markdown(
            """
        Des variables complémentaires ont été créées à partir des données
        existantes pour la visualisation et la modélisation :
        - **`date_et_heure_de_comptage`** : extraction de l'année, mois
          (num/texte), jour (mois/année/semaine - num/texte), heure, semaine
          ISO-8601.
        - **`coordonnees_geographiques`** : extraction de la latitude et
          longitude.
        - **`nom_du_compteur`** : extraction de l'orientation du compteur.
        """
        )

        st.markdown("#### 2.2 Données jours fériés")
        st.markdown(
            """
        Une information sur les jours fériés est récupérée et jointe via une
        API gouvernementale. Ces données sont connues 5 ans à l'avance et sont
        cruciales pour capter leur interaction avec le comportement des
        cyclistes.
        """
        )

        st.markdown("#### 2.3 Données météo")
        st.markdown(
            """
        Des données météorologiques (température, code météo, précipitations,
        neige, altitude) sont intégrées depuis l'API Open-Meteo.com. Elles sont
        essentielles pour la modélisation du trafic à court terme.
        """
        )

        st.markdown("#### 2.4 Données Vélib (envisagé mais non intégré)")
        st.markdown(
            """
        Nous avions envisagé d'utiliser des données des stations Vélib proches
        (nombre de stations ouvertes, bornes disponibles/occupées) via web
        scraping de l'API Vélib' Métropole pour des fins **explicatives
        uniquement**. Cette démarche n'a pas pu être intégrée en raison des
        contraintes de temps et de la nécessité d'un scraping sur une longue
        période.
        """
        )

        st.markdown("#### 2.5 Bilan de l'enrichissement")
        st.markdown(
            """
        Le dataset final, "prêt à l’emploi" pour la modélisation, ne contient
        plus de données manquantes.
        """
        )

    st.markdown("### 3. Visualisations clés")
    st.markdown(
        """
    - **Heures de pointe** : matin (8h–9h) et soir (17h–20h)
    - **Jours de la semaine** : mardi et jeudi > dimanche (trafic utilitaire
      vs loisir)
    - **Top sites** : Sébastopol, Rivoli, Magenta – corridors majeurs
    """
    )

    with st.expander(
        "Détails sur les Transformations et Réduction de Dimension",
        expanded=False
    ):
        st.markdown("#### 3.1 Transformations")
        st.markdown(
            """
        Pour la régression temporelle, une **normalisation des échelles** de
        valeurs des variables quantitatives (ex: coordonnées géographiques)
        est nécessaire. Les variables catégorielles nécessitent un encodage
        numérique. La stratégie dépendra du nombre de valeurs uniques et de
        leur type (ordinale/cardinale), pouvant potentiellement générer de
        nombreuses variables.
        """
        )

        st.markdown("#### 3.2 Réduction de dimension")
        st.markdown(
            """
        Bien que notre jeu de données soit relativement faible en informations
        initiales, l'enrichissement a ajouté des variables qualitatives avec
        de nombreuses catégories (ex: code météo).
        Deux stratégies ont été envisagées :
        - **Encodage de fréquence** pour les catégories à forte cardinalité.
        - **Encodages** réduisant le nombre d'occurrences pour les informations
          géographiques agrégées ou temporelles (sans lien avec la cible :
          Binary ou Frequency Encoding ; avec lien : Target Encoding).
        Une **Analyse en Composantes Principales (PCA)** a été étudiée puis
        écartée en raison du faible nombre de variables explicatives et de la
        nature saisonnière/tendancielle des séries temporelles, qui contient
        déjà de l'information intrinsèque.
        """
        )

    st.info(
        """
    ➡️ Voir les onglets 🔍 *Exploration statistique des données* et 📈
    *Visualisations des données* pour approfondir ces sujets de manière
    interactive"
    """
    )
    st.markdown("### 4. Visualisation et Statistiques")  # Re-numérotation
    with st.expander("📊 4.1 Analyses Univariées", expanded=False):
        st.markdown("#### 4.1.1 Comptage horaire")  # Re-numérotation
        st.markdown(
            """
        La variable **`comptage_horaire`** est fortement concentrée sur les
        faibles valeurs, avec des valeurs extrêmes homogènement distribuées.
        Une valeur aberrante majeure (3070 vélos à 14h le 5 janvier 2025 sur
        "Quai d'Orsay O-E") a été identifiée comme 10 fois supérieure à la
        normale et corrigée.

        Le **test d'Anderson** confirme la **non-normalité** de la
        distribution de `comptage_horaire` pour tous les degrés de tolérance
        (statistique de test = 80094.4).
        """
        )
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            **Analyse du comptage horaire (avant correction)**
            Ce graphique illustre la distribution initiale du comptage horaire
            avec une concentration sur les faibles valeurs et la présence
            d'une valeur aberrante.
            """
            )
        with col2:
            analyse_comptage_horaire = Path("app/assets/analyse_comptage_horaire.png")
            if analyse_comptage_horaire.exists():
                st.image(str(analyse_comptage_horaire), use_container_width=True,
                         caption="Analyse du comptage horaire")
            else:
                st.warning("Image not found: app/assets/analyse_comptage_horaire.png")

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            **Visualisation de la valeur aberrante et sa correction**
            Ce graphique met en évidence la valeur aberrante de 3070 vélos
            sur la tranche horaire de 14h, 5 janvier 2025, sur le compteur
            "Quai d'Orsay O-E", qui a été corrigée car manifestement erronée.
            """
            )
        with col2:
            analyse_valeur_aberrante = Path("app/assets/analyse_valeur_abérrante.png")
            if analyse_valeur_aberrante.exists():
                st.image(str(analyse_valeur_aberrante), use_container_width=True,
                         caption="Analyse de la valeur aberrante")
            else:
                st.warning("Image not found: app/assets/analyse_valeur_abérrante.png")

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            **Nouvelle distribution du comptage horaire après correction**
            Ce graphique montre la distribution ajustée du comptage horaire
            après la correction de la valeur aberrante, offrant une vue
            plus réaliste.
            """
            )
        with col2:
            nouvelle_distribution_comptage_horaire = Path(
                "app/assets/Nouvelle_distribution_comptage_horaire.png"
            )
            if nouvelle_distribution_comptage_horaire.exists():
                st.image(
                    str(nouvelle_distribution_comptage_horaire),
                    use_container_width=True,
                    caption="Nouvelle distribution du comptage horaire"
                )
            else:
                st.warning(
                    "Image not found: app/assets/Nouvelle_distribution_"
                    "comptage_horaire.png"
                )

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            **Nouvelle répartition du comptage horaire (détail)**
            Une vue plus détaillée de la répartition des valeurs après
            correction, confirmant une distribution fortement concentrée sur
            les faibles valeurs.
            """
            )
        with col2:
            nouvelle_repartition_comptage_horaire = Path(
                "app/assets/nouvelle_répartition_comptage_horaire.png"
            )
            if nouvelle_repartition_comptage_horaire.exists():
                st.image(
                    str(nouvelle_repartition_comptage_horaire),
                    use_container_width=True,
                    caption="Nouvelle répartition du comptage horaire"
                )
            else:
                st.warning(
                    "Image not found: app/assets/nouvelle_répartition_"
                    "comptage_horaire.png"
                )

        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            **QQ-plot des résidus du comptage horaire**
            Les données de la variable `comptage_horaire` ne suivent pas une
            loi normale, comme le montre cet écart entre les quantiles
            théoriques d’une loi normale (courbe rouge) et les valeurs
            observées (en bleu). Le test d'Anderson confirme cette
            non-normalité.
            """
            )
        with col2:
            qq_plot_residus = Path("app/assets/QQ_Plot_résidus.png")
            if qq_plot_residus.exists():
                st.image(str(qq_plot_residus), use_container_width=True,
                         caption="QQ-plot des résidus")
            else:
                st.warning("Image not found: app/assets/QQ_Plot_résidus.png")

        st.markdown("#### 4.1.2 Comptage moyen par jour de la semaine")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            Ce graphique met en lumière les différences d’intensité du trafic
            cycliste selon les jours de la semaine, en moyenne horaire.
            🔝 **Jours les plus fréquentés** : Mardi (89.8) et Jeudi (89.6),
            reflétant une forte activité cycliste en milieu de semaine.
            Mercredi (87.9) est proche.
            📉 **Déclin en fin de semaine** : Le trafic baisse légèrement le
            Vendredi (≈81), puis significativement le Samedi (≈62) et surtout
            Dimanche (≈55), montrant une transition vers un usage loisir ou
            un abandon temporaire.
            📆 **Lundi plus faible** : Le Lundi (≈82) est en retrait,
            potentiellement dû à un démarrage de semaine plus progressif.
            """
            )
        with col2:
            comptage_horaire_moyen_par_jour = Path(
                "app/assets/comptage horaire moyen par jour.png"
            )
            if comptage_horaire_moyen_par_jour.exists():
                st.image(
                    str(comptage_horaire_moyen_par_jour), use_container_width=True,
                    caption="Comptage moyen par jour de la semaine"
                )
            else:
                st.warning(
                    "Image not found: app/assets/comptage horaire moyen"
                    " par jour.png"
                )

        st.markdown(
            "#### 4.1.3 Top 10 des heures avec le plus fort comptage "
            "total"
        )
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            Ce graphique met en évidence les dix heures les plus fréquentées
            sur l’ensemble des stations.
            **Heures de pointe très marquées** : La pointe du soir (**18h**)
            est en tête (plus de 7,5 millions), suivie de 19h, 17h et 20h.
            **Heures de début de journée très actives** : Les créneaux 8h et
            9h enregistrent un fort comptage (> 5.9 millions), témoignant des
            déplacements domicile-travail.
            **Heures de milieu de journée plus modérées** : 12h, 13h, 15h et
            16h affichent un comptage moindre (3.6 à 4.1 millions),
            correspondant à des déplacements personnels/professionnels.
            """
            )
        with col2:
            top_10_heures_fort_comptage = Path(
                "app/assets/top_10_heures_fort_comptage.png"
            )
            if top_10_heures_fort_comptage.exists():
                st.image(
                    str(top_10_heures_fort_comptage), use_container_width=True,
                    caption="Top 10 des heures avec le plus fort comptage total"
                )
            else:
                st.warning(
                    "Image not found: app/assets/top_10_heures_fort_comptage.png"
                )

        st.markdown("#### 4.1.4 Top 10 des stations les plus fréquentées")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            Ce graphique met en évidence une forte concentration du trafic
            cycliste sur quelques artères majeures de Paris.
            **Domination du boulevard de Sébastopol et de la rue de Rivoli** :
            Le Totem 73 boulevard de Sébastopol (S-N) domine (plus de 3
            millions), avec deux stations de la rue de Rivoli également
            importantes. Ces itinéraires sont des corridors cyclables majeurs.
            **Importance des compteurs bidirectionnels** : La présence de deux
            directions opposées sur plusieurs stations traduit des flux
            équilibrés.
            **Diversité géographique** : D'autres stations comme Magenta,
            Ménilmontant, Voltaire et la Tournelle suggèrent une diffusion de
            l’usage au-delà de l’hyper-centre.
            """
            )
        with col2:
            top_10_stations = Path("app/assets/Top_10_stations.png")
            if top_10_stations.exists():
                st.image(str(top_10_stations), use_container_width=True,
                         caption="Top 10 des stations les plus fréquentées")
            else:
                st.warning("Image not found: app/assets/Top_10_stations.png")

    with st.expander("🔬 4.2 Analyses Multivariées", expanded=False):
        st.markdown(
            "#### 4.2.1 Corrélation entre cible et variables numériques"
            " (Test de Pearson)"
        )
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            Des corrélations (même faibles) sont observées :
            - La **latitude** du site (c = 0.18) est plus corrélée que la
              longitude (0.02).
            - Les données **météo**, notamment la température (c = 0.16),
              montrent une légère corrélation.
            - Parmi les variables périodiques, l'**heure** (c = 0.28) et le
              **jour de la semaine** (c = -0.10) sont à investiguer pour
              l'analyse de séries temporelles.
            Une attention particulière sera portée à la **multicolinéarité
            potentielle** lors de la modélisation.
            """
            )
        with col2:
            matrice_de_correlation = Path("app/assets/matrice de correlation.png")
            if matrice_de_correlation.exists():
                st.image(str(matrice_de_correlation), use_container_width=True,
                         caption="Matrice de corrélation (Pearson)")
            else:
                st.warning("Image not found: app/assets/matrice_de_correlation.png")

        st.markdown(
            "#### 4.2.2 Corrélation entre cible et variables qualitatives"
            " (Tests ANOVA / Kruskal-Wallis)"
        )
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            Le **test ANOVA** a indiqué un effet significatif des variables
            qualitatives (`arrondissement`, `orientation_compteur`,
            `nom_du_site_de_comptage`, `vacances_scolaires`,
            `weather_code_wmo_code_category`) sur `comptage_horaire`.
            Cependant, les conditions de validité du test (distribution normale
            des résidus, homogénéité des variances via test de Levene) n'étant
            pas respectées, le résultat n'est pas directement utilisable comme
            preuve.

            Nous nous sommes donc rabattus sur le **test non paramétrique de
            Kruskal-Wallis**. Ce test valide qu'il existe des **différences
            significatives** dans la distribution de `comptage_horaire` selon
            les modalités de ces 5 variables, ce qui est également observable
            graphiquement.
            """
            )
        with col2:
            residu_modele_anova = Path("app/assets/résidu_modèle_ANOVA.png")
            if residu_modele_anova.exists():
                st.image(str(residu_modele_anova), use_container_width=True,
                         caption="Résidus du modèle ANOVA")
            else:
                st.warning("Image not found: app/assets/résidu_modèle_ANOVA.png")

            qq_plot_residus_quantiles = Path(
                "app/assets/QQ_plot_résidus_quantiles.png"
            )
            if qq_plot_residus_quantiles.exists():
                st.image(
                    str(qq_plot_residus_quantiles), use_container_width=True,
                    caption="QQ-plot des résidus (quantiles)"
                )
            else:
                st.warning(
                    "Image not found: app/assets/QQ_plot_résidus_quantiles.png"
                )

            test_kruskal_wallis = Path("app/assets/Test Kruskal Wallis.png")
            if test_kruskal_wallis.exists():
                st.image(str(test_kruskal_wallis), use_container_width=True,
                         caption="Test de Kruskal-Wallis")
            else:
                st.warning("Image not found: app/assets/Test Kruskal Wallis.png")

            boxplot_comptage_horaire = Path("app/assets/boxplot_comptage_horaire.png")
            if boxplot_comptage_horaire.exists():
                st.image(str(boxplot_comptage_horaire), use_container_width=True,
                         caption="Boxplot Comptage Horaire")
            else:
                st.warning("Image not found: app/assets/boxplot_comptage_horaire.png")

            boxplot_site_comptage = Path("app/assets/boxplot_site_comptage.png")
            if boxplot_site_comptage.exists():
                st.image(str(boxplot_site_comptage), use_container_width=True,
                         caption="Boxplot Site Comptage")
            else:
                st.warning("Image not found: app/assets/boxplot_site_comptage.png")

            boxplot_vacances_scolaires = Path(
                "app/assets/boxplot_vacances_scolaires.png"
            )
            if boxplot_vacances_scolaires.exists():
                st.image(str(boxplot_vacances_scolaires), use_container_width=True,
                         caption="Boxplot Vacances Scolaires")
            else:
                st.warning("Image not found: app/assets/boxplot_vacances_scolaires.png")

            boxplot_weather = Path("app/assets/Boxplot_Weather.png")
            if boxplot_weather.exists():
                st.image(str(boxplot_weather), use_container_width=True,
                         caption="Boxplot Weather")
            else:
                st.warning("Image not found: app/assets/Boxplot_Weather.png")

        st.markdown("#### 4.2.3 Comptage horaire vs Géolocalisation")
        col1, col2 = st.columns([0.6, 0.4])
        with col1:
            st.markdown(
                """
            Une **disparité significative** du volume total de vélos
            enregistrés par les compteurs est observée entre le centre et la
            périphérie de Paris, ainsi qu'entre le nord-est et le sud-est.
            Cependant, la disparité du maillage des compteurs ne permet pas de
            tirer des conclusions certaines sur la seule base de la
            géolocalisation.
            """
            )
        with col2:
            comptage_total_site = Path("app/assets/comptage_total_site.png")
            if comptage_total_site.exists():
                st.image(str(comptage_total_site), use_container_width=True,
                         caption="Comptage total par site (géolocalisation)")
            else:
                st.warning("Image not found: app/assets/comptage_total_site.png")

with st.expander("🧪 C - Modélisation (10 à 12 min)", expanded=True):
    st.markdown("### 1. Objectif ML : Régression temporelle supervisée")

    st.markdown("""
    - Variable cible : `comptage_horaire`
    - Extraction de variables explicatives additionnelles **spécifiques aux séries
    temporelles**
      - **AR** (*auto-régressives*) et **MM** (*moyennes mobiles*) depuis les valeurs
      précédentes de la **variables cible (prédite ou réelle)**
      - **Oscillatoires** (*de période P*) pour les composantes de l'horodatage suivant
      une période (heure -> 24, jour de la semaine -> 7, etc...)
    """)
    with st.expander("Formalisation mathématique"):
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
    with st.expander("Chronologie de l'étude des modèles"):
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)

        img = Path("app/assets/lineaire.png")
        if img.exists():
            col1.image(str(img), use_container_width=True)
        else:
            col1.warning("Image not found: app/assets/lineaire.png")

        img = Path("app/assets/elasticnet.png")
        if img.exists():
            col2.image(str(img), use_container_width=True)
        else:
            col2.warning("Image not found: app/assets/elasticnet.png")
        img = Path("app/assets/knn.png")

        if img.exists():
            col3.image(str(img), use_container_width=True)
        else:
            col3.warning("Image not found: app/assets/knn.png")

        img = Path("app/assets/random_forest.png")
        if img.exists():
            col4.image(str(img), use_container_width=True)
        else:
            col4.warning("Image not found: app/assets/random_forest.png")

        img = Path("app/assets/xgboost.png")
        if img.exists():
            col5.image(str(img), use_container_width=True)
        else:
            col5.warning("Image not found: app/assets/xgboost.png")

        img = Path("app/assets/sarimax.png")
        if img.exists():
            col6.image(str(img), use_container_width=True)
        else:
            col6.warning("Image not found: app/assets/sarimax.png")

        img = Path("app/assets/deep_learning.png")
        if img.exists():
            col7.image(str(img), use_container_width=True)
        else:
            col7.warning("Image not found: app/assets/deep_learning.png")

    st.markdown("### 2. Performances sans AR/MM")
    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥇 **XGBoost** : meilleur compromis généralisation / précision
    """)
    img = Path("app/assets/xgboost_mean_metrics.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/xgboost_mean_metrics.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥈 **Random Forest** : excellent en entraînement mais problème de généralisation
    """)
    img = Path("app/assets/random_forest_mean_metrics.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/random_forest_mean_metrics.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥉 **KNN** : très bon en entraînement, mais problème de généralisation
    """)
    img = Path("app/assets/knn_mean_metrics.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/knn_mean_metrics.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - ❌ **Régression linéaire** : mauvais en précision (stable en généralisation)
    """)
    img = Path("app/assets/lineaire_mean_metrics.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/lineaire_mean_metrics.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - ❌ **ElasticNet** : amélioration quasi inexistante de la régression linéaire
    malgré une grille de recherche des meilleurs hyperparamètres
    """)
    img = Path("app/assets/elasticnet_mean_metrics.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/elasticnet_mean_metrics.png")

    st.markdown("### 3. Performances explicatives avec AR/MM (avec valeurs réelles"
                " de la cible)")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥇 **Random Forest** : parfait en entraînement et peu de problème de généralisation
    """)
    img = Path("app/assets/random_forest_mean_metrics_armm.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/random_forest_mean_metrics_armm.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥈 **XGBoost** : excellent en entraînement et hyper stable en généralisation
    """)
    img = Path("app/assets/xgboost_mean_metrics_armm.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/xgboost_mean_metrics_armm.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥉 **KNN** : très bon en entraînement, mais problème de généralisation
    """)
    img = Path("app/assets/knn_mean_metrics_armm.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/knn_mean_metrics_armm.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🍫 **Régression linéaire** : honnête en précision (et stable en généralisation)
    """)
    img = Path("app/assets/lineaire_mean_metrics_armm.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/lineaire_mean_metrics_armm.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - ❌ **ElasticNet** : amélioration inexistante de la régression linéaire
    malgré une grille de recherche des meilleurs hyperparamètres
    """)
    img = Path("app/assets/elasticnet_mean_metrics_armm.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/elasticnet_mean_metrics_armm.png")

    st.markdown("### 3. Performance prédictive avec AR/MM en conditions réelles"
                " **step-by-step**")
    st.markdown("""
    Ce mode de prédiction sous entend qu'il n'y a **aucune utilisation des valeurs
    réelles** de la variable cible pour calculer les variables autorégressives ou
    moyennes mobiles, **seules les valeurs prédites** de la variable cible sont
    réutilisées récursivement pour ce calcul (*qui est du coup **bien plus couteux***)
    """)

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥇 **Random Forest** :
    """)
    img = Path("app/assets/random_forest_mean_metrics_forecast.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/"
                     "random_forest_mean_metrics_forecast.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥈 **XGBoost** :
    """)
    img = Path("app/assets/xgboost_mean_metrics_forecast.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/xgboost_mean_metrics_forecast.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - 🥉 **KNN** :
    """)
    img = Path("app/assets/knn_mean_metrics_forecast.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/knn_mean_metrics_forecast.png")

    col1, col2 = st.columns([1, 2])
    col1.markdown("""
    - ❌ **Régression linéaire** : médiocre en généralisation
    """)
    img = Path("app/assets/lineaire_mean_metrics_forecast.png")
    if img.exists():
        col2.image(str(img), use_container_width=True)
    else:
        col2.warning("Image not found: app/assets/lineaire_mean_metrics_forecast.png")

    st.markdown("""
    - ❌ **SARIMAX** : très rapide en calcul de prédiction mais extrêmement couteux
      en entraînement et très faible en généralisation
    """)

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
    - Exploiter les possibilité des réseaux de neurones:
      - profondeur des données (8 ans d’archives disponible sur OpenData Paris)
      - multiplication des sources (un même modèle peut être appliqué à différents
        compteurs).
    - Exploiter la prédiction **multi-canal** des architectures de modèles neuronaux
      (transformation des données pour profiter de la prédiction parallèle du modèle
      Granite TTM)
    """)
    st.info("""
    👈 Les sections suivantes permettent d'explorer nos résultats intéractivement:
    - 🔍 Exploration statistique intéractive des données
    - 📈 Visualisation intéractive des données
    - 🧪 Laboratoire intéractif de modélisation
    """)

st.caption("Projet réalisé dans le cadre de la formation Machine Learning Engineering"
           " - DataScientest avril 2025")
