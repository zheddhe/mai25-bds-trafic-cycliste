import pandas as pd
import numpy as np
import logging
from collections import Counter
import statsmodels.formula.api as smf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Set up logger
logger = logging.getLogger(__name__)


def logit_analysis(data: pd.DataFrame, config: dict, adjustments: dict) -> None:
    """
    Perform a logistic regression using a formula and apply coefficient adjustments.

    Parameters:
    - data: DataFrame containing the dataset
    - config: dict with keys:
        - "target": str, name of the response variable
        - "features": list of str, names of the explanatory variables
    - adjustments: dict where each key is a variable to adjust,
        and each value is a dict with:
        - type: "normalize" or "inverse"
        - range: tuple (min, max) for "normalize"
    """
    # Build regression formula
    target = config["target"]
    features = config["features"]
    formula = f"{target} ~ " + " + ".join(features)

    # Fit logistic model
    model = smf.logit(formula, data=data).fit()

    # Display model summary
    logger.info("\n%s", model.summary2().as_text())

    # Extract coefficients
    params = model.params.copy()

    # Apply adjustments
    for var, adj in adjustments.items():
        if var in params:
            if adj["type"] == "normalize":
                min_val, max_val = adj["range"]
                params[var] /= (max_val - min_val)
            elif adj["type"] == "inverse":
                params[var] = -1 / np.exp(params[var])

    # Compute odds ratios
    odds_ratios = np.exp(params)

    # Format and display
    result_df = pd.DataFrame(odds_ratios, columns=["Odds Ratios"])
    logger.info("\nNormalized Odds Ratios:\n%s", result_df.to_markdown())


def cross_validation_with_resampling(X, y, model):
    """
    Evaluate different resampling strategies using cross-validation.
    """

    resamplers = {
        "SMOTE": SMOTE(k_neighbors=3),
        "Oversampling": RandomOverSampler(sampling_strategy='not majority'),
        "Undersampling": RandomUnderSampler(sampling_strategy='majority'),
    }

    for name, resampler in resamplers.items():
        skf = StratifiedKFold(n_splits=5)
        f1_scores = []

        logger.info(f"=== {name} ===")

        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)

            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_resampled, y_resampled)

            y_pred = model_clone.predict(X_test)
            f1 = f1_score(y_test, y_pred)
            f1_scores.append(f1)

        rounded_scores = [round(score, 5) for score in f1_scores]
        logger.info(f"Scores: {rounded_scores}")
        logger.info(f"Mean F1-score: {np.mean(f1_scores):.5f}\n")


def binarize_proba(probas, threshold):
    return (probas >= threshold).astype(int)


def cross_validation_with_resampling_and_threshold(X, y, model, thresholds=None):
    """
    Cross-validation with resampling and threshold optimization on
    predicted probabilities.
    """

    resamplers = {
        "SMOTE": SMOTE(k_neighbors=3),
        "Oversampling": RandomOverSampler(sampling_strategy='not majority'),
        "Undersampling": RandomUnderSampler(sampling_strategy='majority'),
    }

    if thresholds is None:
        thresholds = np.arange(0.0, 1.0, 0.01)

    for name, resampler in resamplers.items():
        skf = StratifiedKFold(n_splits=5)
        best_thresholds = []
        f1_scores = []

        logger.info(f"=== {name} ===")

        for train_idx, test_idx in skf.split(X, y):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # Dynamically adjust k_neighbors for SMOTE, not needed for other resamplers
            if isinstance(resampler, SMOTE):
                class_counts = Counter(y_train)
                min_class_size = min(class_counts.values())
                k_neighbors = min(5, min_class_size - 1) if min_class_size > 1 else 1
                resampler = SMOTE(k_neighbors=k_neighbors)

            X_resampled, y_resampled = resampler.fit_resample(  # type: ignore
                X_train,
                y_train
            )

            model_clone = model.__class__(**model.get_params())
            model_clone.fit(X_resampled, y_resampled)

            y_proba = model_clone.predict_proba(X_test)[:, 1]
            scores = [f1_score(y_test, binarize_proba(y_proba, t)) for t in thresholds]

            best_idx = np.argmax(scores)  # type: ignore
            best_thresholds.append(thresholds[best_idx])
            f1_scores.append(scores[best_idx])

        logger.info(f"F1 Scores: {[round(s, 2) for s in f1_scores]}")
        logger.info(f"Average threshold = {np.mean(best_thresholds):.3f}")
        logger.info(f"Average F1-score  = {np.mean(f1_scores):.5f}\n")
