import streamlit as st

st.title("🚲 Projet Trafic Cycliste – Démarche & Résultats")

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

    st.markdown("### 2. Feature engineering & enrichissement")
    st.markdown("""
    - Extraction de variables cycliques : heure, jour, mois, etc.
    - Données exogènes enrichies : météo (`Open-Meteo`), jour férié
      et vacances scolaires (API gouvernementale)
    - Variables géographiques (lat/lon), orientation du compteur, etc.
    """)

    st.markdown("### 3. Visualisations clés")
    st.markdown("""
    - **Heures de pointe** : matin (8h–9h) et soir (17h–20h)
    - **Jours de la semaine** : mardi et jeudi > dimanche (trafic utilitaire vs loisir)
    - **Top sites** : Sébastopol, Rivoli, Magenta – corridors majeurs
    """)

    st.info("➡️ Rendez-vous dans l’onglet 📈 *Visualisation et Statistiques*"
            " pour les graphiques intéractifs")

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

    st.info("➡️ Voir l’onglet 🧪 *Évaluation des modèles* pour utiliser notre"
            " laboratoire d'entraînement intéractif")

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
