import os
import streamlit as st
import pandas as pd
import logging
from smartcheck.dataframe_common import (
    # load_dataset_from_config,
    apply_percent_range_selection,
)
from smartcheck.modeling_project_specific import (
    train_timeseries_model,
    compute_metrics,
)
from app.utils.model_logic import (
    run_evaluation_per_compteur,
    display_metrics_table,
    manage_dataset_modeling,
    manage_sidebar_modeling_parameters,
    display_train_parameters,
)

# --- Constants and helpers ---
SITE_LABELS = {
    ('Totem 73 boulevard de Sébastopol', 'S-N'): "Sébastopol_S-N",
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol_N-S",
    ('102 boulevard de Magenta', 'SE-NO'): "Magenta_SE-NO",
    ('Pont de Bercy', 'NE-SO'): "Bercy_NE-SO",
    ('Pont de Bercy', 'SO-NE'): "Bercy_SO-NE",
    ('135 avenue Daumesnil', 'SE-NO'): "Daumesnil_SE-NO",
    ("180 avenue d'Italie", 'N-S'): "Italie_N-S",
    ('27 quai de la Tournelle', 'NO-SE'): "Tournelle_NO-SE",
    ('27 quai de la Tournelle', 'SE-NO'): "Tournelle_SE-NO",
    ('10 avenue de la Grande Armée', 'SE-NO'): "Armée_SE-NO",
    ('10 boulevard Auguste Blanqui', 'NE-SO'): "Blanqui_NE-SO",
    ('106 avenue Denfert Rochereau', 'NE-SO'): "Rochereau_NE-SO",
    ('129 rue Lecourbe', 'SO-NE'): "Lecourbe_SO-NE",
    ('132 rue Lecourbe', 'NE-SO'): "Lecourbe_NE-SO",
    ("147 avenue d'Italie", 'S-N'): "Italie_S-N",
    ('152 boulevard du Montparnasse', 'E-O'): "Montparnasse_E-O",
    ('152 boulevard du Montparnasse', 'O-E'): "Montparnasse_O-E",
    ('16 avenue de la Porte des Ternes', 'E-O'): "Ternes_E-O",
    ('163 boulevard Brune', 'SE-NO'): "Brune_SE-NO",
    ("18 quai de l'Hôtel de Ville", 'NO-SE'): "Ville_NO-SE",
    ("18 quai de l'Hôtel de Ville", 'SE-NO'): "Ville_SE-NO",
    ('21 boulevard Auguste Blanqui', 'SO-NE'): "Blanqui_SO-NE",
    ('24 boulevard Jourdan', 'E-O'): "Jourdan_E-O",
    ('243 boulevard Saint Germain', 'NO-SE'): "Germain_NO-SE",
    ('27 boulevard Davout', 'N-S'): "Davout_N-S",
    ('27 boulevard Diderot', 'E-O'): "Diderot_E-O",
    ('28 boulevard Diderot', 'E-O'): "Diderot_E-O",
    ('28 boulevard Diderot', 'O-E'): "Diderot_O-E",
    ('33 avenue des Champs Elysées', 'NO-SE'): "Elysées_NO-SE",
    ('35 boulevard de Ménilmontant', 'NO-SE'): "Ménilmontant_NO-SE",
    ('36 quai de Grenelle', 'NE-SO'): "Grenelle_NE-SO",
    ('36 quai de Grenelle', 'SO-NE'): "Grenelle_SO-NE",
    ('38 rue Turbigo', 'NE-SO'): "Turbigo_NE-SO",
    ('38 rue Turbigo', 'SO-NE'): "Turbigo_SO-NE",
    ('39 quai François Mauriac', 'NO-SE'): "Mauriac_NO-SE",
    ('39 quai François Mauriac', 'SE-NO'): "Mauriac_SE-NO",
    ('42 boulevard Soult', 'N-S'): "Soult_N-S",
    ('42 boulevard Soult', 'S-N'): "Soult_S-N",
    ('44 avenue des Champs Elysées', 'SE-NO'): "Elysées_SE-NO",
    ('51 boulevard du Général Martial Valin', 'SE-NO'): "Valin_SE-NO",
    ('56 boulevard Kellermann', 'E-O'): "Kellermann_E-O",
    ('6 rue Julia Bartet', 'NE-SO'): "Bartet_NE-SO",
    ('6 rue Julia Bartet', 'SO-NE'): "Bartet_SO-NE",
    ('67 boulevard Voltaire', 'SE-NO'): "Voltaire_SE-NO",
    ('7 avenue de la Grande Armée', 'NO-SE'): "Armée_NO-SE",
    ('72 avenue de Flandre', 'SO-NE'): "Flandre_SO-NE",
    ('72 boulevard Brune', 'NO-SE'): "Brune_NO-SE",
    ('72 boulevard Richard Lenoir', 'S-N'): "Lenoir_S-N",
    ('72 boulevard Voltaire', 'NO-SE'): "Voltaire_NO-SE",
    ('77 boulevard Masséna', 'NE-SO'): "Masséna_NE-SO",
    ('77 boulevard Masséna', 'SO-NE'): "Masséna_SO-NE",
    ('77 boulevard Richard Lenoir', 'N-S'): "Lenoir_N-S",
    ('81 boulevard Mortier', 'N-S'): "Mortier_N-S",
    ('81 boulevard Mortier', 'S-N'): "Mortier_S-N",
    ('87 avenue de Flandre', 'NE-SO'): "Flandre_NE-SO",
    ('89 boulevard de Magenta', 'NO-SE'): "Magenta_NO-SE",
    ('9 boulevard Jourdan', 'O-E'): "Jourdan_O-E",
    ('98 boulevard Poniatowski', 'NE-SO'): "Poniatowski_NE-SO",
    ('98 boulevard Poniatowski', 'SO-NE'): "Poniatowski_SO-NE",
    ("Face 104 rue d'Aubervilliers", 'N-S'): "Aubervilliers_N-S",
    ("Face 104 rue d'Aubervilliers", 'S-N'): "Aubervilliers_S-N",
    ('Face au 16 avenue de la  Porte des Ternes', 'O-E'): "Ternes_O-E",
    ("Face au 25 quai de l'Oise", 'NE-SO'): "Oise_NE-SO",
    ("Face au 25 quai de l'Oise", 'SO-NE'): "Oise_SO-NE",
    ('Face au 4 avenue de la porte de Bagnolet', 'E-O'): "Bagnolet_E-O",
    ('Face au 4 avenue de la porte de Bagnolet', 'O-E'): "Bagnolet_O-E",
    ("Face au 40 quai D'Issy", 'NE-SO'): "Issy_NE-SO",
    ("Face au 40 quai D'Issy", 'SO-NE'): "Issy_SO-NE",
    ('Face au 48 quai de la marne', 'NE-SO'): "Marne_NE-SO",
    ('Face au 48 quai de la marne', 'SO-NE'): "Marne_SO-NE",
    ('Face au 49 boulevard du Général Martial Valin', 'NO-SE'): "Valin_NO-SE",
    ('Face au 70 quai de Bercy', 'N-S'): "Bercy_N-S",
    ('Face au 70 quai de Bercy', 'S-N'): "Bercy_S-N",
    ('Face au 8 avenue de la porte de Charenton', 'NO-SE'): "Charenton_NO-SE",
    ('Face au 8 avenue de la porte de Charenton', 'SE-NO'): "Charenton_SE-NO",
    ('Pont Charles De Gaulle', 'NE-SO'): "Gaulle_NE-SO",
    ('Pont Charles De Gaulle', 'SO-NE'): "Gaulle_SO-NE",
    ('Pont National', 'NE-SO'): "National_NE-SO",
    ('Pont National', 'SO-NE'): "National_SO-NE",
    ('Pont de la Concorde', 'N-S'): "Concorde_N-S",
    ('Pont de la Concorde', 'S-N'): "Concorde_S-N",
    ('Pont des Invalides', 'S-N'): "Invalides_S-N",
    ('Pont des Invalides (couloir bus)', 'N-S'): "Invalides_N-S",
    ('Pont du Garigliano', 'NO-SE'): "Garigliano_NO-SE",
    ('Pont du Garigliano', 'SE-NO'): "Garigliano_SE-NO",
    ("Quai d'Orsay", 'E-O'): "Orsay_E-O",
    ("Quai d'Orsay", 'O-E'): "Orsay_O-E",
    ('Quai des Tuileries', 'NO-SE'): "Tuileries_NO-SE",
    ('Quai des Tuileries', 'SE-NO'): "Tuileries_SE-NO",
    ('Totem 64 Rue de Rivoli', 'E-O'): "Rivoli_E-O",
    ('Totem 64 Rue de Rivoli', 'O-E'): "Rivoli_O-E",
    ("Totem 85 quai d'Austerlitz", 'NO-SE'): "Austerlitz_NO-SE",
    ("Totem 85 quai d'Austerlitz", 'SE-NO'): "Austerlitz_SE-NO",
    ('Totem Cours la Reine', 'E-O'): "Reine_E-O",
    ('Totem Cours la Reine', 'O-E'): "Reine_O-E",
    ('Voie Georges Pompidou', 'NE-SO'): "Pompidou_NE-SO",
    ('Voie Georges Pompidou', 'SO-NE'): "Pompidou_SO-NE",
}
SITE_LABELS_DEFAULT = {
    ('Totem 73 boulevard de Sébastopol', 'S-N'): "Sébastopol_S-N",
    ('Totem 73 boulevard de Sébastopol', 'N-S'): "Sébastopol_N-S",
    ('102 boulevard de Magenta', 'SE-NO'): "Magenta_SE-NO",
    ('Pont de Bercy', 'NE-SO'): "Bercy_NE-SO",
    ('Pont de Bercy', 'SO-NE'): "Bercy_SO-NE",
    ('135 avenue Daumesnil', 'SE-NO'): "Daumesnil_SE-NO",
    ("180 avenue d'Italie", 'N-S'): "Italie_N-S",
    ('27 quai de la Tournelle', 'NO-SE'): "Tournelle_NO-SE",
    ('27 quai de la Tournelle', 'SE-NO'): "Tournelle_SE-NO",
}
EXCLUDED_COLUMNS_DEFAULT = [
    "weather_code_wmo_code",
    "date_et_heure_de_comptage_hour",
    "date_et_heure_de_comptage_day",
    "date_et_heure_de_comptage_day_of_year",
    "date_et_heure_de_comptage_day_of_week",
    "date_et_heure_de_comptage_week",
    "date_et_heure_de_comptage_month",
    "date_et_heure_de_comptage_year",
    "latitude",
    "longitude",
    "arrondissement",
    "elevation",
    "date_et_heure_de_comptage_sin_week",
    "date_et_heure_de_comptage_cos_week",
    "date_et_heure_de_comptage_cos_day_of_year",
    "date_et_heure_de_comptage_sin_day_of_year",
]
AVAILABLE_COLUMNS_TO_EXCLUDE = [
    "weather_code_wmo_code_category",
    "jour_ferie",
    "vacances_scolaires",
    "temperature_2m_c",
    "rain_mm",
    "snowfall_cm",
    "date_et_heure_de_comptage_week_end",
    "date_et_heure_de_comptage_sin_hour",
    "date_et_heure_de_comptage_cos_hour",
    "date_et_heure_de_comptage_sin_day_of_week",
    "date_et_heure_de_comptage_cos_day_of_week",
    "date_et_heure_de_comptage_cos_month",
    "date_et_heure_de_comptage_sin_month",
]
AVAILABLE_COLUMNS = EXCLUDED_COLUMNS_DEFAULT+AVAILABLE_COLUMNS_TO_EXCLUDE
MANDATORY_COLUMNS = [
    "nom_du_site_de_comptage",
    "orientation_compteur",
    "date_et_heure_de_comptage_local",
    "comptage_horaire",
]
AVAILABLE_MODELS = [
    "LinearRegression",
    "KNN",
    "RandomForest",
    "XGBoost",
    "ElasticNet (*)",
]

logger = logging.getLogger(__name__)


@st.cache_data(show_spinner=True)
def cached_train_model(df_compteur,
                       model_type,
                       scaler_type,
                       target_col,
                       drop_columns,
                       temp_feats,
                       test_ratio,
                       forecast):
    if os.environ.get("IS_TESTING") == "1":
        return {
            "y_test": [1, 2],
            "y_test_pred": [1.1, 1.9],
            "X_test_dates": pd.date_range("2025-04-01", periods=2, freq="h")
        }
    return train_timeseries_model(
        df_compteur=df_compteur,
        model_type=model_type,
        scaler_type=scaler_type,
        target_col=target_col,
        drop_columns=drop_columns,
        temp_feats=temp_feats,
        test_ratio=test_ratio,
        forecast=forecast,
    )


st.title("🧪 Laboratoire d'évaluation des modèles")
st.markdown("""
Cette section vous permet de tester différents modèles de régression
sur les données de comptage vélo avec des options personnalisables.
""")
st.info("""
👈 Le dataset est préchargé mais vous pouvez forcer son rechargement via le menu
""")

# --- Enrichissement du menu ---
with st.sidebar:
    df = manage_dataset_modeling(st)
    train_config = manage_sidebar_modeling_parameters(st)

display_train_parameters(train_config, AVAILABLE_COLUMNS, st)

# --- Controles Generaux avant entrainement et rendu
if not train_config["selected_sites"]:
    st.warning("Sélectionnez au moins un compteur à modéliser.")
    st.stop()
if train_config["range"][0] == train_config["range"][1]:
    st.warning("La plage sélectionnée est vide.")
    st.stop()

grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])
with st.spinner("⏳ Entraînement des modèles en cours..."):
    results = {}
    metrics_table = []
    for compteur_id, df_compteur in grouped:
        if compteur_id in train_config["selected_sites"]:
            logger.info(f"Training and prediction started for counter [{compteur_id}]")
            res = cached_train_model(
                df_compteur=apply_percent_range_selection(
                    df_compteur,
                    train_config["range"],
                ),
                model_type=train_config["model"],
                scaler_type=train_config["scaler"],
                target_col="comptage_horaire",
                drop_columns=train_config["drop_cols"],
                temp_feats=[
                    train_config["ar_nb"],
                    train_config["mm_nb"],
                    train_config["mm_season"],
                ],
                test_ratio=1 - train_config["split"],
                forecast=train_config["use_forecast"],
            )
            logger.info(f"Training and prediction done for counter [{compteur_id}]")
            results[compteur_id] = res
            train_metrics = compute_metrics(res["y_train"],
                                            res["y_train_pred"])
            test_metrics = compute_metrics(res["y_test"],
                                           res["y_test_pred"])
            combined_row = {
                "compteur": SITE_LABELS[compteur_id],
                "description": compteur_id,
                "R2_train": train_metrics.get("R2", None),
                "RMSE_train": train_metrics.get("RMSE", None),
                "MAE_train": train_metrics.get("MAE", None),
                "R2_test": test_metrics.get("R2", None),
                "RMSE_test": test_metrics.get("RMSE", None),
                "MAE_test": test_metrics.get("MAE", None),
            }
            metrics_table.append(combined_row)

# --- Synthèse globale des performances ---
if train_config and train_config["show_metrics"] and metrics_table:
    st.markdown("## 🧾 Synthèse des métriques de modélisation")
    display_metrics_table(metrics_table, st_module=st)
    # --- Affichage par compteur ---
    st.markdown("## 🎯 Rapports par compteur")
    run_evaluation_per_compteur(
        results,
        SITE_LABELS,
        train_config["show_metrics"],
        train_config["show_preds"],
        train_config["show_resid"],
        train_config["show_interp"],
        periode_limite=train_config["selected_dates"],
        st_module=st
    )
