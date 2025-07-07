import streamlit as st
from pathlib import Path
from app.config import APP_NAME

# Set path to the sections directory
PAGES_DIR = Path(__file__).parent / "sections"

st.set_page_config(page_title=APP_NAME, layout="wide")

st.sidebar.title("🗂️ Navigation")
pages = {
    "🏠 Introduction au projet": "home",
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
