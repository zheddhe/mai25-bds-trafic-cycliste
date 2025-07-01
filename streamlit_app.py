import streamlit as st
from pathlib import Path

PAGES_DIR = Path(__file__).parent / "app"

st.set_page_config(page_title="Cyclist Traffic Demo", layout="wide")

st.sidebar.title("🗂️ Navigation")
pages = {
    "🏠 Introduction au projet": "home",
    "🔍 Exploration des données": "data_exploration",
    "📈 Visualisation et Statistiques": "data_visualization",
    "🧪 Evaluation des modèles": "modeling"
}

selection = st.sidebar.radio("Go to", list(pages.keys()))
page_file = PAGES_DIR / f"{pages[selection]}.py"

with open(page_file, encoding="utf-8") as f:
    exec(f.read(), globals())
