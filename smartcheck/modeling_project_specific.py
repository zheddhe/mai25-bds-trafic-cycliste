import logging
from typing import Dict, List, Optional, Tuple, cast, Union
from datetime import timedelta
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical, Real
from xgboost import XGBRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from statsmodels.tsa.statespace.sarimax import SARIMAX, SARIMAXResults
from sklearn.preprocessing import (
    OneHotEncoder, MinMaxScaler,
    StandardScaler, RobustScaler
)
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

SEARCH_SPACES_LINEAR = {
    'fit_intercept': Categorical([True, False]),
    'positive': Categorical([False, True]),  # True = contraintes ≥ 0
}
SEARCH_SPACES_ELASTICNET = {
    'alpha': Real(1e-3, 100.0, prior='log-uniform'),
    'l1_ratio': Real(0.1, 1.0)
}
SEARCH_SPACES_KNN = {
    'n_neighbors': Integer(1, 50),  # Taille du voisinage
    'weights': Categorical(['uniform', 'distance']),
    'p': Integer(1, 2),  # Distance : 1 = Manhattan, 2 = Euclidean
}
SEARCH_SPACES_XGB = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 15),
    'learning_rate': Real(1e-3, 0.3, prior='log-uniform'),
    'subsample': Real(0.5, 1.0),
    'colsample_bytree': Real(0.5, 1.0),
    'gamma': Real(0, 10.0),
    'reg_alpha': Real(1e-4, 10.0, prior='log-uniform'),  # L1
    'reg_lambda': Real(1e-4, 10.0, prior='log-uniform'),  # L2
    'min_child_weight': Integer(1, 20),
}
SEARCH_SPACES_RANDOM_FOREST = {
    'n_estimators': Integer(100, 1000),
    'max_depth': Integer(3, 30),
    'min_samples_split': Integer(2, 20),
    'min_samples_leaf': Integer(1, 20),
    'max_features': Categorical(['sqrt', 'log2']),
}

logger = logging.getLogger(__name__)


def auto_adjust_n_iter(search_space: dict, requested_iter: int) -> int:
    total = 1
    for dim in search_space.values():
        if isinstance(dim, Categorical):
            total *= len(dim.categories)
        else:
            # pour Real / Integer : espace infini
            return requested_iter  # on ne limite pas
    return min(requested_iter, total)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred),
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
        start, end = pd.to_datetime(periode_limite).to_pydatetime()
        if start == end:
            end = start + timedelta(days=1)
        ax.set_xlim(start, end)
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
        start, end = pd.to_datetime(periode_limite).to_pydatetime()
        if start == end:
            end = start + timedelta(days=1)
        ax1.set_xlim(start, end)
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
    ax2.set_title(f"Linear trend in residuals – Counter {compteur}")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Residual")
    if periode_limite:
        start, end = pd.to_datetime(periode_limite).to_pydatetime()
        if start == end:
            end = start + timedelta(days=1)
        ax2.set_xlim(start, end)
    ax2.legend()
    ax2.grid(True)

    return fig1, fig2, model.params["t_numeric"]


def interpret_model(
    model_results: dict,
) -> Optional[List[Figure]]:
    """
    Generates a feature importance bar chart for interpretable models,
    including LinearRegression, ElasticNet, RandomForest and XGBoost
    (including when wrapped in BayesSearchCV).

    Returns a list with a single matplotlib Figure showing sorted feature
    importances, or None if the model is not interpretable via coefficients.
    """
    pipe = model_results["pipe"]
    model_figs = []

    reg_step = pipe.named_steps["reg"]

    # === Handle models wrapped in BayesSearchCV ==
    if isinstance(reg_step, BayesSearchCV):
        model = reg_step.best_estimator_  # type: ignore
    else:
        model = reg_step

    # === Try to extract transformed feature names ===
    prep = pipe.named_steps["prep"]
    if hasattr(prep, "get_feature_names_out"):
        features = prep.get_feature_names_out()
    else:
        features = [f"feat_{i}" for i in range(model.n_features_in_)]

    # === Linear models: plot coefficients ===
    if isinstance(model, (LinearRegression, ElasticNet, Ridge, Lasso)):
        coeffs = model.coef_
        importance = pd.Series(coeffs, index=features).sort_values()

        fig, ax = plt.subplots(figsize=(8, 10))
        importance.plot(kind="barh", ax=ax)
        ax.set_title(f"Feature Importance – {type(model).__name__}")
        model_figs.append(fig)
        return model_figs

    # === XGBoost: plot feature importance based on gain ===
    elif isinstance(model, XGBRegressor):
        booster = model.get_booster()
        importance_dict = booster.get_score(importance_type="gain")

        # Map back feature names (XGBoost uses f0, f1, ...)
        importance_series = pd.Series({
            features[int(k[1:])]: v for k, v in importance_dict.items()
        }).sort_values()

        fig, ax = plt.subplots(figsize=(8, 10))
        importance_series.plot(kind="barh", ax=ax)
        ax.set_title("Feature Importance – XGBoost (gain)")
        model_figs.append(fig)
        return model_figs

    # === RandomForest: feature importances ===
    elif isinstance(model, RandomForestRegressor):
        importance = pd.Series(model.feature_importances_, index=features).sort_values()

        fig, ax = plt.subplots(figsize=(8, 10))
        importance.plot(kind="barh", ax=ax)
        ax.set_title("Feature Importance – Random Forest")
        model_figs.append(fig)
        return model_figs

    # === KNN: no native interpretability ===
    elif isinstance(model, KNeighborsRegressor):
        logger.warning("KNN has no native feature importances.")
        return None

    # === No interpretable model found ===
    return None


def train_timeseries_model(
    df_compteur: pd.DataFrame,
    model_type: str,
    scaler_type: str = "",
    target_col: str = "comptage_horaire",
    timestamp_col: str = "date_et_heure_de_comptage",
    drop_columns: Optional[list[str]] = None,
    apply_datetime: bool = True,
    temp_feats: list[int] = [0, 0, 1],
    test_ratio: float = 0.2,
    forecast: bool = True,
    iter_grid_search: int = 0,
) -> dict:
    """
    Full training logic on a single compteur, with optional AR features and split.

    Returns a dict containing trained model, train/test data, and predictions.
    """
    logger.info(f"Train timeseries with [df_len={len(df_compteur)}"
                f" | model_type={model_type}]"
                f" | scaler_type={scaler_type}]"
                f" | drop_columns={drop_columns}]"
                f" | apply_datetime={apply_datetime}]"
                f" | temp_feats={temp_feats}]"
                f" | test_ratio={test_ratio}]"
                f" | forecast={forecast}]")

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

    ar_transformer = None
    if temp_feats[:2] != [0, 0]:
        ar_transformer = AutoregressiveFeaturesTransformer(
            nb_ar=temp_feats[0],
            nb_mm=temp_feats[1],
            roll_wind=temp_feats[2],
        )
        X_train, X_train_dates, y_train = ar_transformer.fit_transform(
            X_train, X_train_dates, y_train
        )
        logger.info(f"AR({temp_feats[0]}) et MM({temp_feats[1]}[{temp_feats[2]}h])"
                    " features are applied on train data")

    numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
    categorical_cols = X_train.select_dtypes(include='object').columns.tolist()

    if scaler_type == "StandardScaler":
        scaler = StandardScaler()
    elif scaler_type == "RobustScaler":
        scaler = RobustScaler()
    else:
        scaler = MinMaxScaler()

    preprocessing = ColumnTransformer([
        ("num", scaler, numeric_cols),
        ("cat", OneHotEncoder(
             handle_unknown="ignore",
             # drop='first',  # avoid multicolinearity but introduce warnings
             sparse_output=False
         ), categorical_cols)
    ])

    if model_type == "KNN":
        model = KNeighborsRegressor(n_jobs=-1)
        search_spaces = SEARCH_SPACES_KNN
    elif model_type == "RandomForest":
        model = RandomForestRegressor(n_jobs=-1, random_state=1)
        search_spaces = SEARCH_SPACES_RANDOM_FOREST
    elif model_type == "XGBoost":
        model = XGBRegressor(random_state=1)
        search_spaces = SEARCH_SPACES_XGB
    elif model_type == "ElasticNet":
        search_spaces = SEARCH_SPACES_ELASTICNET
        model = ElasticNet(max_iter=10000, tol=1e-4, random_state=1)
    else:
        search_spaces = SEARCH_SPACES_LINEAR
        model = LinearRegression(n_jobs=-1)
    if iter_grid_search > 0:
        tscv = TimeSeriesSplit(n_splits=5)
        final_model = BayesSearchCV(
            estimator=model,
            search_spaces=search_spaces,
            cv=tscv,
            n_iter=auto_adjust_n_iter(search_spaces, iter_grid_search),
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            random_state=1
        )
    else:
        final_model = model

    pipe_model = Pipeline([
        ("prep", preprocessing),
        ("reg", final_model)
    ])
    logger.debug(f"Pipeline Model specs used: {pipe_model}")

    pipe_model.fit(X_train, y_train)
    logger.debug("Model training achieved")

    best_params = None
    if iter_grid_search > 0:
        fitted_model = pipe_model.named_steps['reg']
        best_params = fitted_model.best_params_
        logger.info(f"Bayesian grid search best params [{best_params}]")
    y_train_pred = pipe_model.predict(X_train)
    logger.debug("Predictions on Train data achieved")

    if ar_transformer:
        if not forecast:
            X_test, X_test_dates, y_test = ar_transformer.transform_with_known_y(
                X_test, X_test_dates, y_test
            )
            logger.info(f"AR({temp_feats[0]}) et MM({temp_feats[1]}[{temp_feats[2]}h])"
                        " features are applied on test data")
            y_test_pred = pipe_model.predict(X_test)
        else:
            # Assemble full prediction base:
            # - all train rows with y known
            # - all test rows with y unknown (NaN), but features known
            X_full = pd.concat(
                [X_train, X_test],
                ignore_index=True
            )
            y_full = pd.concat(
                [y_train, pd.Series([np.nan] * len(y_test))],
                ignore_index=True
            )
            dates_full = pd.concat(
                [X_train_dates, X_test_dates],
                ignore_index=True
            )
            last_window_df = X_full.copy()
            last_window_df[target_col] = y_full
            last_window_df[timestamp_col] = dates_full
            logger.info(f"recursive predict on an horizon of {len(y_test)} hour(s)")
            y_test_pred = recursive_forecast_model(
                pipe_model,
                ar_transformer,
                last_window_df=last_window_df,
                horizon=len(y_test),
                target_col=target_col
            )
    else:
        y_test_pred = pipe_model.predict(X_test)
    logger.debug("Predictions on Test data achieved")

    return {
        "timestamp_col": timestamp_col,
        "target_col": target_col,
        "ar_transformer": ar_transformer,
        "pipe": pipe_model,
        "best_params": best_params,
        "X_train": X_train,
        "X_train_dates": X_train_dates,
        "X_test": X_test,
        "X_test_dates": X_test_dates,
        "y_train": y_train,
        "y_train_pred": y_train_pred,
        "y_test": y_test,
        "y_test_pred": y_test_pred,
    }


class SARIMAXWrapper(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        order=(1, 0, 0),
        seasonal_order=(0, 0, 0, 0),
        trend=None,
        use_exo=True
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.trend = trend
        self.use_exo = use_exo

    def fit(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y: Optional[Union[np.ndarray, pd.Series]] = None
    ):
        if y is None:
            raise ValueError(
                "y (endogenous variable) must be provided for SARIMAX."
            )

        exog = X if self.use_exo else None
        logger.info("Fitting SARIMAX model.")
        self.model_ = SARIMAX(
            endog=y,
            exog=exog,
            order=self.order,
            seasonal_order=self.seasonal_order,
            trend=self.trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self.results_ = cast(SARIMAXResults, self.model_.fit(disp=False))
        self.n_train_ = len(y)
        logger.info("SARIMAX model fitted successfully.")
        return self

    def predict(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        n_periods: Optional[int] = None
    ) -> np.ndarray:
        if not hasattr(self, "results_"):
            raise ValueError("Model must be fitted before calling predict.")

        exog = X if self.use_exo else None
        if n_periods is None:
            if X is not None:
                n_periods = len(X)
            else:
                raise ValueError(
                    "You must provide either X or n_periods for forecasting."
                )

        start = self.n_train_
        end = start + n_periods - 1
        logger.info("Predicting future values (out-of-sample).")
        return self.results_.predict(start=start, end=end, exog=exog)

    def predict_in_sample(self) -> np.ndarray:
        if not hasattr(self, "results_"):
            raise ValueError("Model must be fitted before calling predict.")
        logger.info("Returning in-sample fitted values.")
        return self.results_.fittedvalues  # type: ignore


def recursive_forecast_model(
    pipe: Pipeline,
    ar_transformer: AutoregressiveFeaturesTransformer,
    last_window_df: pd.DataFrame,
    horizon: int,
    target_col: str
) -> List[float]:
    """
    Efficient recursive forecast using AR/MM features and exogenous inputs.

    Args:
        pipe (Pipeline): Trained pipeline (prep + regressor).
        ar_transformer: Fitted AutoregressiveFeaturesTransformer.
        last_window_df (pd.DataFrame): Full base with historical train + test X.
        horizon (int): Number of future steps to forecast.
        timestamp_col (str): Name of datetime column.
        target_col (str): Name of target variable.

    Returns:
        List[float]: Forecasted target values (one per horizon step).
    """
    future_preds = []
    current_df = last_window_df.copy()

    # Prepare history buffer (NumPy for speed)
    required_lag = max(
        ar_transformer.nb_ar,
        ar_transformer.nb_mm * ar_transformer.roll_wind
    )
    recent_y = np.array(
        current_df[target_col].dropna().values, dtype=np.float32
    )

    if len(recent_y) < required_lag:
        raise ValueError(
            f"Insufficient history: need at least {required_lag} values in "
            f"`recent_y`, but only {len(recent_y)} provided."
        )

    # Pre-buffer exogenous features (sans target)
    steps_to_forecast = current_df[current_df[target_col].isna()]
    exog_features = steps_to_forecast.drop(columns=[target_col])
    exog_features = exog_features.reset_index(drop=True)
    forecast_rows = []

    for i in range(horizon):
        exog_row = exog_features.iloc[[i]].copy()

        try:
            X_next = ar_transformer.transform_recursive_step(
                exog_row, recent_y.tolist()
            )
        except Exception as e:
            logger.warning(f"[STEP {i}] Failed to create AR features: {e}")
            break

        X_next_prepped = pipe.named_steps["prep"].transform(X_next)
        y_pred = pipe.named_steps["reg"].predict(X_next_prepped)[0]

        future_preds.append(y_pred)
        recent_y = np.append(recent_y, y_pred)

        exog_row[target_col] = y_pred
        forecast_rows.append(exog_row)

    return future_preds
