from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import (
    f1_score, mean_squared_error
)
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
import logging

# Set up logger
logger = logging.getLogger(__name__)


def compare_search_methods(model_name, model, param_grid, X_train, X_test,
                           y_train, y_test, results: dict):
    """
    Compare different hyperparameter search strategies
    for both classification and regression models.
    """

    if is_classifier(model):
        scoring = 'f1'  # or 'accuracy', depending on preference

        def metric_func(y_true, y_pred):  # type: ignore
            return f1_score(y_true, y_pred)

        metric_name = "f1_score"

    elif is_regressor(model):
        scoring = 'neg_mean_squared_error'

        def metric_func(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)

        metric_name = "mean_squared_error"

    else:
        raise ValueError("The provided model is neither a classifier nor a regressor.")

    search_methods = {
        'GridSearchCV': GridSearchCV(
            estimator=model, param_grid=param_grid,
            cv=5, scoring=scoring
        ),
        'RandomizedSearchCV': RandomizedSearchCV(
            estimator=model, param_distributions=param_grid,
            n_iter=5, cv=5, scoring=scoring, random_state=42
        ),
        'BayesSearchCV': BayesSearchCV(
            estimator=model, search_spaces=param_grid,
            n_iter=5, cv=5, scoring=scoring, random_state=42
        )
    }

    results[model_name] = {}
    logger.info(f"### Model: {model_name} ###")

    for search_name, search in search_methods.items():
        logger.info(f"== Method: {search_name} ==")

        try:
            search.fit(X_train, y_train)
        except Exception as e:
            logger.error(f"Error during search with {search_name}: {e}")
            continue

        best_params = search.best_params_
        best_score = search.best_score_

        try:
            y_pred = search.predict(X_test)
            test_metric = metric_func(y_test, y_pred)
        except NotFittedError:
            logger.warning(f"Model not fitted with {search_name}")
            test_metric = None

        results[model_name][search_name] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            f'test_{metric_name}': test_metric
        }

        logger.info(f"Best parameters ({search_name}): {best_params}")
        logger.info(f"CV score ({scoring}): {best_score:.4f}")
        if test_metric is not None:
            logger.info(f"Test score ({metric_name}): {test_metric:.4f}\n")
        else:
            logger.info(f"Test score ({metric_name}): None\n")
