import logging
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shap
import streamlit as st
import statsmodels.api as sm
from matplotlib.figure import Figure
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score,
)
from smartcheck.dataframe_project_specific import train_test_split_time_aware
from smartcheck.preprocessing_project_specific import (
    DatetimePeriodicsTransformer,
    AutoregressiveFeaturesTransformer,
)


logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "r2": r2_score(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }


def plot_predictions(
    compteur: str,
    dates: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    periode_limite: Optional[Tuple[str, str]] = None,
) -> Figure:
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(
        dates["date_et_heure_de_comptage_local"], y_true, label="True Values"
    )
    ax.plot(
        dates["date_et_heure_de_comptage_local"],
        y_pred,
        label="Predictions",
        linestyle="--",
    )
    ax.set_title(f"Predictions – Counter {compteur}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Hourly Count")
    if periode_limite:
        ax.set_xlim(*pd.to_datetime(periode_limite).to_pydatetime())
    ax.legend()
    ax.grid(True)
    return fig


def compute_residuals_plot(
    compteur: str,
    dates: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    periode_limite: Optional[Tuple[str, str]] = None,
) -> Tuple[Figure, Figure, float]:
    df = dates.copy()
    df["date"] = df["date_et_heure_de_comptage_local"]
    df["y_true"] = y_true.values
    df["y_pred"] = y_pred
    df["residuals"] = df["y_true"] - df["y_pred"]
    df = df.sort_values("date")

    # Residuals with rolling mean
    fig1, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(df["date"], df["residuals"], label="Residuals", alpha=0.3)
    ax1.plot(
        df["date"],
        df["residuals"].rolling(window=24, center=True).mean(),
        label="Rolling Mean (24)",
        linewidth=2,
    )
    ax1.axhline(0, color="black", linestyle="--", linewidth=1)
    ax1.set_title(f"Residuals over time – Counter {compteur}")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Residual")
    if periode_limite:
        ax1.set_xlim(*pd.to_datetime(periode_limite).to_pydatetime())
    ax1.legend()
    ax1.grid(True)

    # Linear trend
    df["t_numeric"] = (df["date"] - df["date"].min()).dt.total_seconds() / 86400
    X = sm.add_constant(df["t_numeric"])
    y = df["residuals"]
    model = sm.OLS(y, X).fit()
    df["trend"] = model.predict(X)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(df["date"], df["residuals"], label="Residuals", alpha=0.4)
    ax2.plot(df["date"], df["trend"], label="Linear Trend", color="red")
    ax2.axhline(0, color="black", linestyle="--", linewidth=1)
    ax2.set_title("Linear trend in residuals")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual")
    if periode_limite:
        ax2.set_xlim(*pd.to_datetime(periode_limite).to_pydatetime())
    ax2.legend()
    ax2.grid(True)

    return fig1, fig2, model.params["t_numeric"]


def interpret_model(
    compteur: str,
    model_results: dict,
) -> Optional[List[Figure]]:
    pipe = model_results["pipe"]
    X_test = model_results["X_test"]
    y_pred = model_results["y_test_pred"]

    model_figs = []
    for _, step in pipe.named_steps.items():
        if isinstance(step, LinearRegression):
            features = pipe.named_steps["prep"].get_feature_names_out()
            coeffs = step.coef_
            importance = pd.Series(coeffs, index=features).sort_values()
            fig, ax = plt.subplots(figsize=(8, 10))
            importance.plot(kind="barh", ax=ax)
            ax.set_title("Feature Importance – Linear Model")
            model_figs.append(fig)
            return model_figs

        if isinstance(step, KNeighborsRegressor):
            pipe_input = pipe.named_steps["prep"]
            X_transformed = pipe_input.transform(X_test)
            pca = PCA(n_components=2)
            X_proj = pca.fit_transform(X_transformed)

            if pca.explained_variance_ratio_.sum() < 0.9:
                logger.warning("PCA explained variance < 90%%, skipping plot.")
                return None

            coeffs = pca.components_.T
            df_plot = pd.DataFrame({
                "PC1": X_proj[:, 0],
                "PC2": X_proj[:, 1],
                "prediction": y_pred,
            })

            fig, ax = plt.subplots(figsize=(10, 8))
            sns.scatterplot(
                x="PC1", y="PC2", hue="prediction",
                data=df_plot, palette="coolwarm", ax=ax
            )
            for i in range(X_test.shape[1]):
                ax.arrow(0, 0, coeffs[i, 0]*1.5, coeffs[i, 1]*1.5,
                         color="black", alpha=0.5, head_width=0.01)
            ax.set_title(f"PCA – Counter {compteur}")
            model_figs.append(fig)
            return model_figs

    return None


def get_shap_background(X: pd.DataFrame, method: str, k: int = 100) -> pd.DataFrame:
    """
    Generate background data for SHAP explanation.

    Args:
        X (pd.DataFrame): Training feature set.
        method (str): One of "sample", "tail", or "kmeans".
        k (int): Number of samples or clusters.

    Returns:
        pd.DataFrame: Background sample for SHAP.
    """
    if method == "sample":
        return shap.sample(X, k)
    elif method == "tail":
        return X.tail(k)
    elif method == "kmeans":
        dense = shap.kmeans(X, k)
        return pd.DataFrame(dense.data, columns=X.columns)
    else:
        raise ValueError(f"Unknown background method: {method}")


def render_shap_summary_streamlit(
    pipe,
    X: pd.DataFrame,
    background_method: str = "sample",
    background_size: int = 100,
    nb_samples: int = 50,
    max_display: int = 10,
    show: bool = True
) -> None:
    """
    Display SHAP summary plot in Streamlit for a sklearn Pipeline.

    Args:
        pipe: A fitted sklearn pipeline with "prep" and "reg" steps.
        X (pd.DataFrame): Full feature DataFrame (before transform).
        background_method (str): 'sample', 'tail', or 'kmeans'.
        background_size (int): Number of background samples.
        max_display (int): Max features to display in plot.
        show (bool): Whether to display plot in Streamlit.

    Returns:
        None
    """
    # === Transform X ===
    X_transformed = pipe.named_steps["prep"].transform(X)
    feature_names = pipe.named_steps["prep"].get_feature_names_out()
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    # === Background selection ===
    background = get_shap_background(X_df, method=background_method,
                                     k=background_size)

    # === Select SHAP explainer ===
    reg = pipe.named_steps["reg"]
    model_name = reg.__class__.__name__.lower()
    if "xgb" in model_name or "tree" in model_name or "forest" in model_name:
        explainer = shap.Explainer(reg, background)
    elif "linear" in model_name:
        explainer = shap.LinearExplainer(reg, background)
    else:
        explainer = shap.KernelExplainer(reg.predict, background)

    # === Compute SHAP values ===
    X_subset = X_df.sample(nb_samples, random_state=42)
    shap_values = explainer(X_subset)

    # === Plot summary ===
    shap.summary_plot(shap_values,
                      features=X_df,
                      feature_names=feature_names,
                      max_display=max_display,
                      plot_type="bar")

    if show:
        fig = plt.gcf()
        st.pyplot(fig)
        plt.clf()


def train_timeseries_model(
    df_compteur: pd.DataFrame,
    model_type: str,
    target_col: str = "comptage_horaire",
    timestamp_col: str = "date_et_heure_de_comptage",
    rolling_window: int = 24,
    drop_columns: Optional[list[str]] = None,
    apply_datetime: bool = True,
    use_ar1_ma24: bool = True,
    test_ratio: float = 0.2,
) -> dict:
    """
    Full training logic on a single compteur, with optional AR features and split.

    Returns a dict containing trained model, train/test data, and predictions.
    """
    df = df_compteur.copy()

    if apply_datetime:
        tr_date = DatetimePeriodicsTransformer(
            timestamp_col=timestamp_col
        )
        df = tr_date.transform(df)
        timestamp_col = timestamp_col+"_local"

    df = df.sort_values(timestamp_col)

    if drop_columns:
        df = df.drop(columns=[col for col in drop_columns if col in df.columns])

    df = df.sort_values(timestamp_col)

    X_train, X_train_dates, X_test, X_test_dates, y_train, y_test = (
        train_test_split_time_aware(
            df,
            timestamp_cols=[timestamp_col],
            target_col=target_col,
            test_size=test_ratio,
        )
    )

    if use_ar1_ma24:
        ar_transformer = AutoregressiveFeaturesTransformer(
            rolling_window=rolling_window
        )
        X_train, X_train_dates, y_train = ar_transformer.fit_transform(
            X_train, X_train_dates, y_train
        )
        X_test, X_test_dates, y_test = ar_transformer.transform_test(
            X_test, X_test_dates, y_test
        )

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    preprocessing = ColumnTransformer([
        ("num", MinMaxScaler(), numeric_cols),
        ("cat",
         OneHotEncoder(
             handle_unknown="ignore",
             # drop='first',  # avoid multicolinearity
             sparse_output=False
         ),
         categorical_cols)
    ])

    if model_type == "KNN":
        model = KNeighborsRegressor(n_jobs=-1)
    else:
        model = LinearRegression()

    pipe_model = Pipeline([
        ("prep", preprocessing),
        ("reg", model)
    ])

    pipe_model.fit(X_train, y_train)
    y_test_pred = pipe_model.predict(X_test)

    return {
        "pipe": pipe_model,
        "X_train": X_train,
        "X_train_dates": X_train_dates,
        "X_test": X_test,
        "X_test_dates": X_test_dates,
        "y_train": y_train,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }
