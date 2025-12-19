import os
import pandas as pd
import numpy as np
import streamlit as st
import logging
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
from smartcheck.modeling_project_specific import (
    compute_metrics,
    plot_predictions,
    compute_residuals_plot,
    interpret_model,
    train_timeseries_model,
)
from smartcheck.dataframe_common import (
    load_dataset_from_config,
    apply_percent_range_selection,
)

logger = logging.getLogger(__name__)

# --- Constants and helpers ---
DATASET_NAME = "velo_comptage_ml_ready_data"
DEFAULT_TEST_PERIOD = ('2025-04-01', '2025-04-15')
MAX_TEST_PERIOD = ('2024-03-01', '2025-04-15')
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
ID_COLUMNS = [
    "nom_du_site_de_comptage",
    "orientation_compteur",
]
MANDATORY_COLUMNS = [
    "date_et_heure_de_comptage_local",
    "comptage_horaire",
]
AVAILABLE_MODELS = [
    "XGBoost",
    "RandomForest",
    "KNN",
    "LinearRegression",
    "ElasticNet",
]
ITER_GRID_SEARCH = 25


@st.cache_data(show_spinner=True)
def cached_load_dataset_ml(uploaded_file):
    if os.environ.get("IS_TESTING") == "1":
        return pd.DataFrame(columns=["nom_du_site_de_comptage",
                                     "orientation_compteur", "comptage_horaire"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file, sep=",", index_col=0)
    else:
        df = load_dataset_from_config(DATASET_NAME, sep=",", index_col=0)
    return df


def manage_dataset_modeling(st_module=None) -> pd.DataFrame:
    st = st_module or __import__("streamlit")

    uploaded_file = st.file_uploader(
        "Personnaliser le dataset",
        type=["csv"],
        accept_multiple_files=False,
        label_visibility="collapsed",
    )
    if st.button("🔁 Rechargement du Dataset"):
        cached_load_dataset_ml.clear()  # type: ignore
        cached_train_model.clear()  # type: ignore
        st.rerun()

    with st.spinner("⏳ Chargement du dataset en cours..."):
        df = cached_load_dataset_ml(uploaded_file)
    source = "(personnalisées)" if uploaded_file else "(originales)"
    if df is not None and isinstance(df, pd.DataFrame):
        st.success(f"✅ Données {source} chargées avec succès.")
    else:
        st.error(f"❌ Données {source} non chargée.")
        st.stop()

    if df is None or not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()

    return df


@st.cache_data(show_spinner=True)
def cached_train_model(df_compteur,
                       model_type,
                       scaler_type,
                       target_col,
                       drop_columns,
                       temp_feats,
                       test_ratio,
                       forecast,
                       use_gridsearch):
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
        iter_grid_search=ITER_GRID_SEARCH if use_gridsearch else 0,
    )


def build_param_table(
    best_params: dict,
    min_params: dict,
    max_params: dict
) -> pd.DataFrame:
    """
    Format parameter search results into a comparative table.

    Returns:
        pd.DataFrame: DataFrame with rows: minimum, best, maximum
    """
    df = pd.DataFrame.from_dict(
        {
            "minimum": min_params,
            "MEILLEUR": best_params,
            "maximum": max_params
        },
        orient="index"
    )
    df.index.name = "Paramètre"
    return df


def display_report_per_counter(results, train_config, st_module=None):

    st = st_module or __import__("streamlit")

    for compteur_id, res in results.items():
        label = SITE_LABELS[compteur_id]
        with st.expander(f"📉 Rapport pour {label}"):

            if train_config["show_metrics"]:
                st.markdown("### 📈 Métriques")
                counter_metrics_table = []
                train_metrics = compute_metrics(res["y_train"], res["y_train_pred"])
                test_metrics = compute_metrics(res["y_test"], res["y_test_pred"])
                train_dates = res.get(
                    "X_train_dates"
                ).get("date_et_heure_de_comptage_local")
                combined_train = {
                    "Plage": "Train",
                    "Début période": train_dates.min(),
                    "Fin période": train_dates.max(),
                    "Nb échantillons": train_dates.count(),
                    "R²": train_metrics.get("R2", None),
                    "RMSE": train_metrics.get("RMSE", None),
                    "MAE": train_metrics.get("MAE", None),
                }
                counter_metrics_table.append(combined_train)
                test_dates = res.get(
                    "X_test_dates"
                ).get("date_et_heure_de_comptage_local")
                combined_test = {
                    "Plage": "Test",
                    "Début période": test_dates.min(),
                    "Fin période": test_dates.max(),
                    "Nb échantillons": test_dates.count(),
                    "R²": test_metrics.get("R2", None),
                    "RMSE": test_metrics.get("RMSE", None),
                    "MAE": test_metrics.get("MAE", None),
                }
                counter_metrics_table.append(combined_test)
                display_counter_metrics_table(
                    counter_metrics_table,
                    params=res["params"],
                    st_module=st
                )

            if train_config["show_preds"]:
                st.markdown("### 🔮 Prédictions")
                fig = plot_predictions(
                    compteur=label,
                    dates=res["X_test_dates"],
                    y_true=res["y_test"],
                    y_pred=res["y_test_pred"],
                    periode_limite=train_config["selected_dates"],
                )
                st.pyplot(fig)
                plt.close(fig)

            if train_config["show_resid"]:
                st.markdown("### 🧾 Résidus")
                fig1, fig2, slope = compute_residuals_plot(
                    compteur=label,
                    dates=res["X_test_dates"],
                    y_true=res["y_test"],
                    y_pred=res["y_test_pred"],
                    periode_limite=train_config["selected_dates"],
                )
                st.pyplot(fig1)
                plt.close(fig1)
                st.info(f"**Pente** de la **dérive** des résidus: [{slope:.4f}]")
                st.pyplot(fig2)
                plt.close(fig2)

            if train_config["show_interp"]:
                st.markdown("### 🧠 Interprétation")
                interp_figs = interpret_model(res)
                if interp_figs:
                    for fig in interp_figs:
                        st.pyplot(fig)
                        plt.close(fig)


def display_counter_metrics_table(
    counter_metrics_table,
    params=None,
    st_module=None
):
    st = st_module or __import__("streamlit")

    if not counter_metrics_table:
        return

    df_metrics = pd.DataFrame(counter_metrics_table)
    styled_df = (
        df_metrics.style
        .format(precision=3)
        .background_gradient(
            subset=["R²"],
            cmap="RdYlGn",  # green = good, red = bad
            vmin=0.0,
            vmax=1.0,
        )
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    if params:
        df_params = df_params = build_param_table(
            params["best_params"],
            params["min_params"],
            params["max_params"]
        )
        st.info("Résultat de la recherche Bayesienne des **Meilleurs paramètres**")
        st.dataframe(df_params, use_container_width=True, hide_index=False)


def display_global_metrics_table(metrics_table, st_module=None, show_mean=True):
    st = st_module or __import__("streamlit")

    if not metrics_table:
        return

    df_metrics = pd.DataFrame(metrics_table)
    styled_df = (
        df_metrics.style
        .format(precision=3)
        .background_gradient(
            subset=["R² train", "R² test"],
            cmap="RdYlGn",  # green = good, red = bad
            vmin=0.0,
            vmax=1.0,
        )
    )
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    if show_mean:
        # display averages
        num_cols = df_metrics.select_dtypes(include=np.number).columns
        mean_row = df_metrics[num_cols].mean().to_frame().T
        mean_row.insert(0, "Nombre de compteurs pour les métriques moyennes",
                        df_metrics.shape[0])

        styled_mean = (
            mean_row.style
            .format(precision=4)
            .background_gradient(
                subset=["R² train", "R² test"],
                cmap="RdYlGn", vmin=0.0, vmax=1.0,
            )
        )

        st.dataframe(styled_mean, use_container_width=True, hide_index=True)


def display_train_parameters(train_config, st_module=None):
    st = st_module or __import__("streamlit")

    with st.expander("Résumé des paramètres d'entrainement courants",
                     expanded=False):
        col1, col2 = st.columns([1, 1])
        portion = train_config['range'][1] - train_config['range'][0]
        with col1:
            st.markdown(f"""
            - **Modèle utilisé** : `{train_config['model']}`
            - **Recherche Bayesienne des hyperparamètres** :
            `{train_config['use_gridsearch']}`
            - **Mise à l'échelle utilisé** : `{train_config['scaler']}`
            - **Nb d'Auto-régressives** : `{train_config['ar_nb']}`
            - **Nb de Moyennes mobiles** : `{train_config['mm_nb']}`
            - **Taille fenêtre mobile (heures)** : `{train_config['mm_season']}`
            - **Calcul avec valeurs prédites des AR/MM** :
            `{train_config['use_forecast']}`
            - **Portion du dataset d'origine** : `{portion}%`
            (entre `{train_config['range'][0]}%` et
            `{train_config['range'][1]}%`)
            - **Répartition Train/Test** :
            `{int(train_config['split'] * 100)}%` /
            `{int((1 - train_config['split']) * 100)}%`
            - **Plage d'affichage des prédictions** :
            `{train_config['selected_dates'][0].strftime("%Y-%m-%d")}` →
            `{train_config['selected_dates'][1].strftime("%Y-%m-%d")}`
            - **Afficher les métriques** : `{train_config['show_metrics']}`
            - **Afficher les prédictions** : `{train_config['show_preds']}`
            - **Afficher les résidus et la pente** : `{train_config['show_resid']}`
            - **Afficher l'interprétation** : `{train_config['show_interp']}`
            """)
        with col2:
            df_cols_exclues = pd.DataFrame({
                "Variables Explicatives": AVAILABLE_COLUMNS,
                "Exclue": [
                    col in train_config["drop_cols"]
                    for col in AVAILABLE_COLUMNS
                ]
            })
            st.data_editor(
                df_cols_exclues,
                column_config={
                    "Exclue": st.column_config.CheckboxColumn("Exclue",
                                                              disabled=True)
                },
                hide_index=True,
                use_container_width=True,
                disabled=True,
                num_rows="dynamic"
            )

    # Controls of the parameters
    if not train_config["selected_sites"]:
        st.warning("Sélectionnez au moins un compteur à modéliser.")
        st.stop()
    if train_config["range"][0] == train_config["range"][1]:
        st.warning("La plage sélectionnée est vide.")
        st.stop()


def manage_sidebar_modeling_parameters(st_module=None) -> Dict:
    st = st_module or __import__("streamlit")

    st.header("🔧 Paramètres")

    # --- Sélection des compteurs ---
    label_options = list(SITE_LABELS.values())
    label_default_options = list(SITE_LABELS_DEFAULT.values())
    selected_labels = st.multiselect(
        "🎯 Choix des compteurs à modéliser",
        label_options,
        default=label_default_options
    )
    selected_sites = [k for k, v in SITE_LABELS.items() if v in selected_labels]

    # --- Sélection des modèles ---
    with st.expander("Relatifs à la **Modélisation**", expanded=False):
        model = st.radio("Type de modèle", AVAILABLE_MODELS, key="model_rad")
        use_gridsearch = st.checkbox(
            "Recherche Bayesienne des hyperparamètres (*)",
            value=False,
            key="use_gridsearch_cb"
        )
        st.info("(*) La recherche est potentiellement très **longue**")

    with st.expander("Relatifs au **Preprocessing**", expanded=False):
        # --- Sélection du scaler ---
        scaler = st.radio(
            "Mise à l'échelle",
            ("MinMaxScaler", "StandardScaler", "RobustScaler"),
            key="scaler_rad"
        )

        # --- Echantillonnage du dataset + répartition train/test ---
        range = st.slider(
            "Portion du dataset d'origine à utiliser", 0.0, 100.0,
            (0.0, 100.0), 0.1, format="%.1f %%", key="range_sld"
        )
        split = st.slider(
            "Répartition Train/Test", 0.1, 0.9, 0.75, 0.05,
            key="split_sld"
        )

        # --- Sélection des variables explicatives ---
        with st.expander("❌ **Variables explicatives à exclure**"):
            df_checkbox = pd.DataFrame({
                "Variable": AVAILABLE_COLUMNS,
                "Exclue": [
                    col in EXCLUDED_COLUMNS_DEFAULT
                    for col in AVAILABLE_COLUMNS
                ]
            })
            st.markdown("🔒 Les colonnes suivantes sont obligatoires:")
            st.code("\n".join(MANDATORY_COLUMNS+ID_COLUMNS), language="markdown")
            edited_df = st.data_editor(
                df_checkbox,
                column_config={
                    "Exclue": st.column_config.CheckboxColumn("Exclue")
                },
                hide_index=True,
                use_container_width=True,
                num_rows="fixed",
                key="edited_df_de"
            )
            drop_cols = edited_df.loc[edited_df["Exclue"], "Variable"].tolist()
            drop_cols = drop_cols+ID_COLUMNS

    # --- Sélection des paramètres d'auto regression/moyenne mobile ---
    with st.expander("Relatifs aux **Séries temporelles**", expanded=False):
        ar_nb = st.slider(
            "Nombre de variables **A**uto-**R**égressives (**AR**)", 0, 7, 0, 1,
            key="ar_nb_sld"
        )
        mm_nb = st.slider(
            "Nombre de variables **M**oyennes **M**obiles (**MM**)", 0, 7, 0, 1,
            key="ar_mm_nb"
        )
        mm_season = st.number_input(
            "Taille de fenêtre pour moyennes mobiles (1 lag = 1 heure)",
            min_value=2, max_value=24*7,
            value=24, key="mm_season_inp"
        )
        use_forecast = st.checkbox(
            "Recalcul récursif des AR/MM en prédiction (*)",
            value=False,
            key="use_forecast_cb"
        )
        st.info("(*) Ce mode de calcul peut être très **long**")

    # --- Option des rapports ---
    with st.expander("📊 Option des rapports", expanded=False):
        col_date_min, col_date_max = st.columns(2)
        with col_date_min:
            start_date = st.date_input("Début de l'affichage",
                                       value=DEFAULT_TEST_PERIOD[0],
                                       min_value=MAX_TEST_PERIOD[0],
                                       max_value=MAX_TEST_PERIOD[1],
                                       key="col_date_min_di")
        with col_date_max:
            end_date = st.date_input("Fin de l'affichage",
                                     value=DEFAULT_TEST_PERIOD[1],
                                     min_value=MAX_TEST_PERIOD[0],
                                     max_value=MAX_TEST_PERIOD[1],
                                     key="col_date_max_di")
        selected_dates = (start_date, end_date)
        show_metrics = st.checkbox("Afficher métriques", value=True,
                                   key="show_metrics_cb")
        show_preds = st.checkbox("Afficher prédictions", value=True,
                                 key="show_preds_cb")
        show_resid = st.checkbox("Afficher résidus", value=True,
                                 key="show_resid_cb")
        show_interp = st.checkbox("Afficher interprétation", value=True,
                                  key="show_interp_cb")

    train_config = {
        "model": model,
        "use_gridsearch": use_gridsearch,
        "scaler": scaler,
        "ar_nb": ar_nb,
        "mm_nb": mm_nb,
        "mm_season": mm_season,
        "use_forecast": use_forecast,
        "range": range,
        "split": split,
        "drop_cols": drop_cols.copy(),
        "selected_dates": selected_dates,
        "selected_sites": selected_sites,
        "show_metrics": show_metrics,
        "show_preds": show_preds,
        "show_resid": show_resid,
        "show_interp": show_interp,
    }

    return train_config


def manage_training(train_config, df) -> Tuple[Dict, List]:
    grouped = df.groupby(["nom_du_site_de_comptage", "orientation_compteur"])
    results = {}
    metrics_table = []
    for compteur_id, df_compteur in grouped:
        if compteur_id in train_config["selected_sites"]:
            df_cpt_ranged = apply_percent_range_selection(
                df_compteur,
                train_config["range"],
            )
            logger.info(f"Training and prediction started for counter [{compteur_id}]")
            res = cached_train_model(
                df_compteur=df_cpt_ranged,
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
                use_gridsearch=train_config["use_gridsearch"],
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
                "R² train": train_metrics.get("R2", None),
                "RMSE train": train_metrics.get("RMSE", None),
                "MAE train": train_metrics.get("MAE", None),
                "R² test": test_metrics.get("R2", None),
                "RMSE test": test_metrics.get("RMSE", None),
                "MAE test": test_metrics.get("MAE", None),
            }
            metrics_table.append(combined_row)

    return results, metrics_table
