from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from skopt import BayesSearchCV
from sklearn.metrics import (
    f1_score, mean_squared_error
)
from sklearn.base import is_classifier, is_regressor
from sklearn.exceptions import NotFittedError
import logging

# Set up logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def compare_search_methods(model_name, model, param_grid, X_train, y_train,
                           X_test, y_test, results: dict):
    """
    Compare différentes méthodes de recherche d'hyperparamètres
    pour la classification ou la régression.
    """

    if is_classifier(model):
        scoring = 'f1'  # ou 'accuracy' selon préférence

        def metric_func(y_true, y_pred):
            return f1_score(y_true, y_pred)
        metric_name = "f1_score"
    elif is_regressor(model):
        scoring = 'neg_mean_squared_error'

        def metric_func(y_true, y_pred):
            return mean_squared_error(y_true, y_pred)
        metric_name = "mean_squared_error"
    else:
        raise ValueError("Le modèle fourni n'est ni un classifieur ni un régressseur.")

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

    for search_name, search in search_methods.items():
        log.info(f"== Méthode : {search_name} ==")

        try:
            search.fit(X_train, y_train)
        except Exception as e:
            log.error(f"Erreur lors de la recherche avec {search_name} : {e}")
            continue

        best_params = search.best_params_
        best_score = search.best_score_

        try:
            y_pred = search.predict(X_test)
            test_metric = metric_func(y_test, y_pred)
        except NotFittedError:
            log.warning(f"Modèle non entraîné avec {search_name}")
            test_metric = None

        results[model_name][search_name] = {
            'best_params': best_params,
            'best_cv_score': best_score,
            f'test_{metric_name}': test_metric
        }

        log.info(f"Meilleurs paramètres ({search_name}): {best_params}")
        log.info(f"Score CV ({scoring}): {best_score:.4f}")
        if test_metric is not None:
            log.info(f"Score test ({metric_name}): {test_metric:.4f}\n")
        else:
            log.info(f"Score test ({metric_name}): None\n")
