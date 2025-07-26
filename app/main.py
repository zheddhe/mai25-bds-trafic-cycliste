import streamlit as st
import logging
from app.config import APP_NAME, PAGES_DIR
from smartcheck.logger_config import setup_logger
setup_logger(logging.INFO)

st.set_page_config(page_title=APP_NAME, layout="wide")

st.sidebar.title("🗂️ Navigation")
pages = {
    "🏠 Présentation de l'application": "home",
    "⚙️ Démarche projet et résultats": "project_presentation",
    "🔍 Exploration statistique des données": "data_exploration",
    "📈 Visualisation graphique des données": "data_visualization",
    "🧪 Laboratoire de modélisation intéractive": "modeling"
}

selection = st.sidebar.radio("Changer de section", list(pages.keys()))
page_file = PAGES_DIR / f"{pages[selection]}.py"

if page_file.exists():
    with open(page_file, encoding="utf-8") as f:
        code = f.read()
    exec(code, globals())
else:
    st.error(f"❌ Page '{pages[selection]}' not found at {page_file}")
