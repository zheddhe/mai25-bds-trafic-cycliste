import streamlit as st

URL_DATASCIENTEST = "https://www.linkedin.com/school/datascientest"
URL_LINKEDIN_REMY = "https://www.linkedin.com/in/remycanal/"
URL_LINKEDIN_ELIAS = "https://www.linkedin.com/in/elias-djouadi/"

st.title("🚲 Application Trafic Cycliste")

st.markdown('''
Bienvenue sur l'application de démonstration de l'étude réalisée lors de la formation
**Data Scientist** de [DataScientest]({URL_DATASCIENTEST}) (promotion bootcamp Mai 2025)
dans le cadre du projet **Trafic Cycliste** (identifié `mai25-bds-trafic-cycliste`).
''')

st.info(f'''
**Auteurs**
- Rémy CANAL ([LinkedIn]({URL_LINKEDIN_REMY}))
- Elias DJOUADI ([LinkedIn]({URL_LINKEDIN_ELIAS}))
''')

st.info('''
👈 Utilisez le **menu de navigation** latéral pour **explorer** :
- La présentation de la démarche projet ainsi que nos résultats et conclusions
- L'exploration intéractive des statistiques après la phase de préparation
  des données
- La visualisation graphique et intéractive des données préparées pour la
  modélisation
- Un laboratoire intéractif de modélisation et d'évaluation des modèles dans
  différentes conditions
''')
