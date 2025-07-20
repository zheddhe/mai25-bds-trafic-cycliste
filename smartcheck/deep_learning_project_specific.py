import os
import logging
import yaml
import re
import joblib
import pandas as pd
from typing import Tuple
from tsfm_public.toolkit import TimeSeriesPreprocessor
from sklearn.preprocessing import OrdinalEncoder
from transformers import (
    Trainer  # type: ignore
)
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


def df_train_test_split_time_aware(
    df: pd.DataFrame,
    timestamp_column: str,
    test_size: float = 0.2,
    sort: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chronological train/test split for a dataframe.

    Args:
        df: DataFrame containing all data.
        timestamp_col: Columns related to time (timezone aware).
        test_size: Fraction of data to use for testing.

    Returns:
        df_train, df_test
    """
    df = df.copy()

    if sort:
        df = df.sort_values(by=timestamp_column)

    # Chronological split
    n_test = int(len(df) * test_size)
    df_train = df[:-n_test]
    df_test = df[-n_test:]

    return df_train, df_test


class SafeTimeSeriesPreprocessorOrdinal(TimeSeriesPreprocessor):
    def _train_categorical_encoder(self, df: pd.DataFrame) -> None:
        cols_to_encode = self._get_columns_to_encode()
        if cols_to_encode:
            df = self._replace_unknowns(df, cols_to_encode, force_include=True)
            self.categorical_encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=999,
            )
            self.categorical_encoder.fit(df[cols_to_encode])

            if logging.getLogger().isEnabledFor(logging.DEBUG):
                for i, col in enumerate(cols_to_encode):
                    cats = self.categorical_encoder.categories_[i]
                    assert "__UNKNOWN__" in cats, (  # type: ignore
                        f"__UNKNOWN__ missing from {col}"
                    )

    def _replace_unknowns(
        self,
        df: pd.DataFrame,
        cols: list[str],
        unknown_token: str = "__UNKNOWN__",
        force_include: bool = False,
    ) -> pd.DataFrame:
        df = df.copy()
        for i, col in enumerate(cols):
            if col not in df.columns:
                continue  # ignore missing
            df[col] = df[col].fillna(unknown_token)

            if self.categorical_encoder and hasattr(
                self.categorical_encoder, "categories_"
            ):
                known = set(self.categorical_encoder.categories_[i])  # type: ignore
            else:
                known = set(df[col].unique())

            df[col] = df[col].apply(
                lambda x: x if x in known else unknown_token
            )

            if force_include and unknown_token not in df[col].values:
                df.loc[df.index[-1], col] = unknown_token
        return df

    def _process_encoding(self, df: pd.DataFrame) -> pd.DataFrame:
        cols_to_encode = self._get_columns_to_encode()
        if self.encode_categorical and cols_to_encode:
            if not self.categorical_encoder:
                raise RuntimeError("Encoder not trained")
            df = self._replace_unknowns(df, cols_to_encode)
            df[cols_to_encode] = self.categorical_encoder.transform(
                df[cols_to_encode]
            )
        return df


def save_preprocessor_state(tsp: SafeTimeSeriesPreprocessorOrdinal, save_dir: str):
    """
    Save the TimeSeriesPreprocessor configuration and encoder.

    Args:
        tsp: The fitted TimeSeriesPreprocessor.
        save_dir: Directory where to save config and encoder.
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, "preprocessor_config.yaml")
    encoder_path = os.path.join(save_dir, "categorical_encoder.joblib")
    scaler_path = os.path.join(save_dir, "scaler_dict.joblib")
    target_scaler_path = os.path.join(save_dir, "target_scaler_dict.joblib")

    config = {
        "timestamp_column": tsp.timestamp_column,
        "id_columns": tsp.id_columns,
        "target_columns": tsp.target_columns,
        "categorical_columns": tsp.categorical_columns,
        "static_categorical_columns": tsp.static_categorical_columns,
        "observable_columns": tsp.observable_columns,
        "control_columns": tsp.control_columns,
        "conditional_columns": tsp.conditional_columns,
        "context_length": tsp.context_length,
        "prediction_length": tsp.prediction_length,
        "scaling": tsp.scaling,
        "scaler_type": tsp.scaler_type,
        "encode_categorical": tsp.encode_categorical,
        "freq": tsp.freq,
    }

    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)

    joblib.dump(tsp.categorical_encoder, encoder_path)
    joblib.dump(tsp.scaler_dict, scaler_path)
    joblib.dump(tsp.target_scaler_dict, target_scaler_path)


def load_preprocessor_state(load_dir: str) -> SafeTimeSeriesPreprocessorOrdinal:
    """
    Load a previously saved TimeSeriesPreprocessor.

    Args:
        load_dir: Directory containing config and encoder.

    Returns:
        A TimeSeriesPreprocessor with fitted encoder.
    """
    config_path = os.path.join(load_dir, "preprocessor_config.yaml")
    encoder_path = os.path.join(load_dir, "categorical_encoder.joblib")
    scaler_path = os.path.join(load_dir, "scaler_dict.joblib")
    target_scaler_path = os.path.join(load_dir, "target_scaler_dict.joblib")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Restore artifacts
    tsp = SafeTimeSeriesPreprocessorOrdinal(**config)
    tsp.categorical_encoder = joblib.load(encoder_path)
    tsp.scaler_dict = joblib.load(scaler_path)
    tsp.target_scaler_dict = joblib.load(target_scaler_path)

    # Assertion on critical items
    assert tsp.timestamp_column is not None
    assert tsp.id_columns is not None
    assert tsp.target_columns is not None

    logging.info(f"📥 Chargement TimeSeries Preprocessor depuis {load_dir}")

    return tsp


def save_granite_model(experiment, model_results: dict, save_dir: str = "."):
    exp_params = model_results["exp_params"]
    best_checkpoint = model_results["best_checkpoint"]
    name = exp_params["name"].replace(" ", "-")
    start = exp_params["sub_range"][0]
    sub_range = exp_params["sub_range"]
    stop = sub_range[1] - 1 if len(sub_range) > 1 else "end"

    filename = (
        f"granite_results_{best_checkpoint}"
        f"_{name}_{start}-{stop}.joblib"
    )
    filepath = f"{save_dir}/{filename}"
    joblib.dump(model_results, filepath)

    logging.info(f"{experiment} saved in {filepath}")


def train_or_resume(
    trainer: Trainer,
    exp_params: dict = {}
) -> str:
    if hasattr(exp_params, "best_checkpoint_dir"):
        checkpoint_dir = exp_params["best_checkpoint_dir"]
    elif hasattr(exp_params, "last_checkpoint_dir"):
        checkpoint_dir = exp_params["last_checkpoint_dir"]
    else:
        checkpoint_dir = ""
    logging.info(f"Checkpoint_dir identified : [{checkpoint_dir}]")
    if os.path.isdir(checkpoint_dir):
        model_weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        logging.info(f"🔁 Loading weights only from {model_weights_path}")
        state_dict = load_file(model_weights_path)
        trainer.model.load_state_dict(state_dict)  # type: ignore
        trainer.train()  # ⚠ repart from scratch mais avec les bons poids
    else:
        logging.info("🆕 Starting training from scratch")
        trainer.train()
    logging.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
    match = re.search(r"checkpoint-\d+",
                      trainer.state.best_model_checkpoint)  # type: ignore
    if match:
        best_checkpoint = match.group()
        print(f"✅ Checkpoint trouvé : {best_checkpoint}")
    else:
        best_checkpoint = ""
        print("❌ Aucun checkpoint trouvé dans la chaîne.")
    return best_checkpoint


def load_model_from_checkpoint(model_class, checkpoint_dir: str, device="cpu"):
    """
    Recharge un modèle Hugging Face depuis un checkpoint avec uniquement les poids.
    - model_class: classe du modèle (ex: TinyTimeMixerForPrediction)
    - checkpoint_dir: dossier contenant model.safetensors et config.json
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    weights_path = os.path.join(checkpoint_dir, "model.safetensors")

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Fichier de poids introuvable : {weights_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Fichier de configuration introuvable : {config_path}")

    logging.info(f"📥 Chargement config du modèle depuis {config_path}")
    model = model_class.from_pretrained(checkpoint_dir, local_files_only=True)

    logging.info(f"🔁 Chargement poids du modèle uniquement depuis {weights_path}")
    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict)

    return model
