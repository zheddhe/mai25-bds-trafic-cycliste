import streamlit as st
import logging
from app.config import APP_NAME, PAGES_DIR
from smartcheck.logger_config import setup_logger
setup_logger(logging.INFO)

st.set_page_config(page_title=APP_NAME, layout="wide")

st.sidebar.title("🗂️ Navigation")
pages = {
    "🏠 Introduction au projet": "home",
    "⚙️ Démarche projet et résultats": "project_presentation",
    "🔍 Exploration des données": "data_exploration",
    "📈 Visualisation et Statistiques": "data_visualization",
    "🧪 Évaluation des modèles": "modeling"
}

selection = st.sidebar.radio("Changer de page", list(pages.keys()))
page_file = PAGES_DIR / f"{pages[selection]}.py"

if page_file.exists():
    with open(page_file, encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
else:
    st.error(f"❌ Page '{pages[selection]}' not found at {page_file}")
