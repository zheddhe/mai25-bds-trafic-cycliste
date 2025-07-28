import streamlit as st
from pathlib import Path

st.title("⚙️ Projet Trafic Cycliste – Démarche & Résultats")

image_projet_mle_path = Path("app/assets/image_projet_mle.png")
if image_projet_mle_path.exists():
    st.image(str(image_projet_mle_path), use_container_width=True)
else:
    st.warning("Image not found: app/assets/image_projet_mle.png")
st.title("⚙️ Projet Trafic Cycliste – Démarche & Résultats")

st.markdown("**Présentation synthétique (20 minutes)**")
with st.expander("🎙️ A - Introduction (1 min)", expanded=False):
    st.markdown("""
    La Ville de Paris dispose de compteurs permanents pour évaluer la pratique cycliste.
    Le but du projet : **prédire l’évolution horaire du trafic vélo** par site, pour :
    - Adapter les aménagements cyclables
    - Détecter les zones à surveiller
    - Évaluer l’impact des facteurs météo, jour férié, horaire, etc.

    👥 **Équipe** : Rémy Canal, Elias Djouadi, (Raphaël Parmentier)

    🧠 **Méthodologie** : mise en place d'une pipeline ML(Ops) complète + laboratoire
    interactif Streamlit
    """)

with st.expander("🔍 B - Exploration & Visualisation (8 min)", expanded=False):
    st.markdown("### 1. Données sources & nettoyages")
    st.markdown("""
    - 940k+ observations, 13 mois glissants – source OpenData Paris
    - Nettoyage des doublons/valeurs manquantes (clusters par nom de compteur)
    - **Reconstruction automatique** de noms de compteur erronés
    """)

    with st.expander("Détails sur le Nettoyage", expanded=False):
        st.markdown("#### 1.1 Traitement des données manquantes")
        st.markdown("""
        Les valeurs manquantes sont regroupées sur des plages d'index contiguës,
        principalement pour des colonnes liées aux photos et identifiants techniques :
        `lien_vers_photo_du_site_de_comptage`, `identifiant_technique_compteur`,
        `id_photos`, `test_lien_vers_photos_du_site_de_comptage`, `id_photo_1`,
        `type_dimage`.

        D'autres clusters d'absence touchent aussi :
        `identifiant_du_compteur`, `identifiant_du_site_de_comptage`,
        `nom_du_site_de_comptage`, `date_d_installation_du_site_de_comptage`,
        `coordonnees_geographiques`, `url_sites`.

        Seuls quelques noms de compteur concentrent l'ensemble de ces observations
        manquantes (ex: '10 avenue de la Grande Armée...', '27 quai de la Tournelle...',
        etc.). Ces noms étaient des valeurs erronées transitoires ; ils ont été inférés
        et recoupés avec la base `Comptage vélo - Compteurs`.

        Après reconduction des données d'origine, seules les données techniques
        spécifiques de site (URL/préfixe/suffixe/id) subsistent. Ces variables,
        jugées peu explicatives, ont été abandonnées.
        """)
        # Pour les "vues condensées" (graphiques), vous devriez les insérer ici
        # via st.image() ou un autre composant Streamlit
        # #si vous avez les fichiers image.
        # Exemple: st.image("vue_condensee_initiale.png",
        # #caption="Vue condensée de la répartition de l’absence de valeur (initiale)")

        st.markdown("#### 1.2 Traitement des types")
        st.markdown("""
        Nous avons ajusté le type de certaines variables et extrait des informations
        périodiques (année, mois, jour, heure, semaine) de la variable
        `date_et_heure_de_comptage`. Des informations géographiques (latitude,longitude)
        ont été extraites de `coordonnees_geographiques` et l'orientation du compteur
        du `nom_du_compteur`.
        """)

    st.markdown("### 2. Feature engineering & enrichissement")
    with st.expander("Détails sur l'Enrichissement", expanded=False):
        st.markdown("#### 2.1 Données intrinsèques / Recombinaisons")
        st.markdown("""
        Des variables complémentaires ont été créées à partir des données existantes
        pour la visualisation et la modélisation :
        - **`date_et_heure_de_comptage`** : extraction de l'année, mois (num/texte),
          jour (mois/année/semaine - num/texte), heure, semaine ISO-8601.
        - **`coordonnees_geographiques`** : extraction de la latitude et longitude.
        - **`nom_du_compteur`** : extraction de l'orientation du compteur.
        """)

        st.markdown("#### 2.2 Données jours fériés")
        st.markdown("""
        Une information sur les jours fériés est récupérée et jointe via une API
        gouvernementale. Ces données sont connues 5 ans à l'avance et sont cruciales
        pour capter leur interaction avec le comportement des cyclistes.
        """)

        st.markdown("#### 2.3 Données météo")
        st.markdown("""
        Des données météorologiques (température, code météo, précipitations, neige,
        altitude) sont intégrées depuis l'API Open-Meteo.com. Elles sont essentielles
        pour la modélisation du trafic à court terme.
        """)

        st.markdown("#### 2.4 Données Vélib (envisagé mais non intégré)")
        st.markdown("""
        Nous avions envisagé d'utiliser des données des stations Vélib proches (nombre
        de stations ouvertes, bornes disponibles/occupées) via web scraping de l'API
        Vélib' Métropole pour des fins **explicatives uniquement**. Cette démarche n'a
        pas pu être intégrée en raison des contraintes de temps et de la nécessité
        d'un scraping sur une longue période.
        """)

        st.markdown("#### 2.5 Bilan de l'enrichissement")
        st.markdown("""
        Le dataset final, "prêt à l’emploi" pour la modélisation, ne contient plus de
        données manquantes.
        """)

    st.markdown("### 3. Visualisations clés")
    st.markdown("""
    - **Heures de pointe** : matin (8h–9h) et soir (17h–20h)
    - **Jours de la semaine** : mardi et jeudi > dimanche (trafic utilitaire vs loisir)
    - **Top sites** : Sébastopol, Rivoli, Magenta – corridors majeurs
    """)

    with st.expander("Détails sur les Transformations et Réduction de Dimension",
                     expanded=False):
        st.markdown("#### 3.1 Transformations")
        st.markdown("""
        Pour la régression temporelle, une **normalisation des échelles** de valeurs
        des variables quantitatives (ex: coordonnées géographiques) est nécessaire.
        Les variables catégorielles nécessitent un encodage numérique. La stratégie
        dépendra du nombre de valeurs uniques et de leur type (ordinale/cardinale),
        pouvant potentiellement générer de nombreuses variables.
        """)

        st.markdown("#### 3.2 Réduction de dimension")
        st.markdown("""
        Bien que notre jeu de données soit relativement faible en informations
        initiales, l'enrichissement a ajouté des variables qualitatives avec de
        nombreuses catégories (ex: code météo).
        Deux stratégies ont été envisagées :
        - **Encodage de fréquence** pour les catégories à forte cardinalité.
        - **Encodages** réduisant le nombre d'occurrences pour les informations
          géographiques agrégées ou temporelles (sans lien avec la cible : Binary ou
          Frequency Encoding ; avec lien : Target Encoding).
        Une **Analyse en Composantes Principales (PCA)** a été étudiée puis écartée
        en raison du faible nombre de variables explicatives et de la nature
        saisonnière/tendancielle des séries temporelles, qui contient déjà de
        l'information intrinsèque.
        """)

    st.info("""
    ➡️ Voir les onglets 🔍 *Exploration statistique des données* et 📈 *Visualisations
    des données* pour approfondir ces sujets de manière intéractive"
    """)

    # --- Nouvelle section pour la Visualisation ---
    st.markdown("## 📈 Visualisation et Statistiques")

    with st.expander("📊 Analyses Univariées", expanded=True): # J'ai mis expanded=True ici pour que cette section soit visible par défaut
        st.markdown("### 1. Comptage horaire")
        st.markdown("""
        La variable **`comptage_horaire`** est fortement concentrée sur les faibles valeurs,
        avec des valeurs extrêmes homogènement distribuées. Une valeur aberrante majeure
        (3070 vélos à 14h le 5 janvier 2025 sur "Quai d'Orsay O-E") a été identifiée
        comme 10 fois supérieure à la normale et corrigée.

        Le **test d'Anderson** confirme la **non-normalité** de la distribution de
        `comptage_horaire` pour tous les degrés de tolérance (statistique de test = 80094.4).
        """)
        # Ici, vous inséreriez les graphiques mentionnés :
        # - Graphique de distribution avec valeur aberrante
        # - QQ-plot pour la normalité

        st.markdown("#### Comptage moyen par jour de la semaine")
        st.markdown("""
        Les jours les plus fréquentés sont le **Mardi (89.8)** et le **Jeudi (89.6)**,
        reflétant une forte activité cycliste en milieu de semaine. Le trafic baisse
        significativement en fin de semaine, avec les valeurs les plus faibles le
        **Samedi (≈62)** et le **Dimanche (≈55)**, indiquant un usage plus loisir
        que utilitaire. Le **Lundi (≈82)** est légèrement en retrait.
        """)
        # st.image("comptage_moyen_jour_semaine.png", caption="Comptage moyen par jour de la semaine")

        st.markdown("#### Top 10 des heures avec le plus fort comptage total")
        st.markdown("""
        Les **heures de pointe** sont très marquées :
        - La pointe du soir (**18h**) est en tête (plus de 7,5 millions de passages cumulés),
          suivie de 19h, 17h, et 20h.
        - La matinée est également très active (**8h** et **9h**), avec plus de 6 et 5,9
          millions de passages respectivement, typiques des déplacements domicile-travail.
        - Les heures de milieu de journée (12h, 13h, 15h, 16h) sont plus modérées.
        """)
        # st.image("top_10_heures.png", caption="Top 10 des heures avec le plus fort comptage total")

        st.markdown("#### Top 10 des stations les plus fréquentées")
        st.markdown("""
        Le **Totem 73 boulevard de Sébastopol (S-N)** domine le classement (plus de 3 millions),
        suivi par deux stations de la **rue de Rivoli**. Ces itinéraires sont des corridors
        cyclables majeurs. La présence de **flux bidirectionnels** sur plusieurs stations
        (ex: S-N et N-S) est importante. D'autres stations comme Magenta, Ménilmontant,
        Voltaire et la Tournelle montrent une diffusion de l'usage au-delà de l'hyper-centre.
        """)
        # st.image("top_10_stations.png", caption="Top 10 des stations les plus fréquentées")

    with st.expander("🔬 Analyses Multivariées", expanded=False):
        st.markdown("### 1. Corrélation entre cible et variables numériques (Test de Pearson)")
        st.markdown("""
        Des corrélations (même faibles) sont observées :
        - La **latitude** du site (c = 0.18) est plus corrélée que la longitude (0.02).
        - Les données **météo**, notamment la température (c = 0.16), montrent une légère
          corrélation.
        - Parmi les variables périodiques, l'**heure** (c = 0.28) et le **jour de la semaine**
          (c = -0.10) sont à investiguer pour l'analyse de séries temporelles.
        Une attention particulière sera portée à la **multicolinéarité potentielle** lors
        de la modélisation.
        """)
        # st.image("heatmap_correlations_pearson.png", caption="Matrice de corrélation (Pearson)")

        st.markdown("### 2. Corrélation entre cible et variables qualitatives (Tests ANOVA / Kruskal-Wallis)")
        st.markdown("""
        Le **test ANOVA** a indiqué un effet significatif des variables qualitatives
        (`arrondissement`, `orientation_compteur`, `nom_du_site_de_comptage`,
        `vacances_scolaires`, `weather_code_wmo_code_category`) sur `comptage_horaire`.
        Cependant, les conditions de validité du test (distribution normale des résidus,
        homogénéité des variances via test de Levene) n'étant pas respectées, le résultat
        n'est pas directement utilisable comme preuve.

        Nous nous sommes donc rabattus sur le **test non paramétrique de Kruskal-Wallis**.
        Ce test valide qu'il existe des **différences significatives** dans la distribution
        de `comptage_horaire` selon les modalités de ces 5 variables, ce qui est
        également observable graphiquement.
        """)
        # st.image("boxplots_qualitatives_cible.png", caption="Distribution du comptage horaire par modalités de variables qualitatives")


        st.markdown("### 3. Comptage horaire vs Géolocalisation")
        st.markdown("""
        Une **disparité significative** du volume total de vélos enregistrés par les
        compteurs est observée entre le centre et la périphérie de Paris, ainsi qu'entre
        le nord-est et le sud-est. Cependant, la disparité du maillage des compteurs
        ne permet pas de tirer des conclusions certaines sur la seule base de la
        géolocalisation.
        """)
        # st.map() ou st.pydeck_chart() avec les données de comptage
        # st.image("carte_comptage_geoloc.png", caption="Comptage total par géolocalisation")


    st.info("➡️ Voir l’onglet 🧪 *Évaluation des modèles* pour utiliser notre"
                " laboratoire d'entraînement intéractif")


with st.expander("🧪 C - Modélisation (10 à 12 min)", expanded=False):
    st.markdown("### 1. Objectif ML : Régression temporelle supervisée")
    st.markdown("""
    - Variable cible : `comptage_horaire`
    - Modèles testés : linéaire, ElasticNet, KNN, RandomForest, XGBoost, SARIMAX,
      Deep Learning
    - Enrichissement AR/MM (auto-régressions, moyennes mobiles) pour capter la
      dynamique horaire saisonnière
    """)

    st.markdown("### 2. Résultats comparés (mode explicatif)")
    st.markdown("""
    - 🥇 **XGBoost** : meilleur compromis généralisation / précision (R² jusqu’à 0.95)
    - 🥈 **Random Forest** : très bon mais parfois instable
    - ➕ **KNN** : bon en moyenne, sensible au bruit
    - ➕ **ElasticNet** : bonne sélection des variables explicatives
    """)

    st.markdown("### 3. Résultats (mode prédictif réel – step-by-step)")
    st.markdown("""
    - ❌ Modèles ML classiques très dégradés en prédiction aveugle
    - ✅ **SARIMAX** : rapide mais peu généralisable
    - 🧠 **Deep Learning (Granite TTM)** :
      - excellent en *zero-shot*
      - améliorable par *fine-tuning* (R² ≈ 0.87 en test sur 1 semaine glissante)
    """)

    st.info("""
    ➡️ Voir l’onglet 🧪 *Laboratoire intéractif de modélisation* pour utiliser
    notre laboratoire d'entraînement et d'observation de la performance des modèles
    en pilotant les conditions
    """)

with st.expander("🔚 D - Conclusion & Ouverture (1 min)", expanded=False):
    st.markdown("""
    ✅ Une approche complète mêlant statistique, modélisation ML, séries temporelles
        et deep learning.

    🚀 Prochaine étape : entraînement multi-compteurs sur 8 ans d’archives ParisData.

    🔍 Exploiter les séries longues pour **entraîner de meilleurs modèles neuronaux**.

    👉 Continuez avec les pages suivantes pour explorer plus intéractivement
        nos résultats :
    - "🔍 Exploration des données"
    - "📈 Visualisation et Statistiques"
    - "🧪 Évaluation des modèles"
    """)

st.caption("Projet réalisé dans le cadre de la formation Machine Learning Engineering"
             " - DataScientest avril 2025")
