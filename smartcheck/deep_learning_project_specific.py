import os
import logging
import yaml
import re
import joblib
import pprint
import pandas as pd
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
import netron
from typing import Tuple, Dict, Any
from tsfm_public.toolkit.time_series_preprocessor import (
    TimeSeriesPreprocessor,
    TYPE_TO_STRING,
)
from tsfm_public import (
    TinyTimeMixerForPrediction,
    TrackingCallback,
    count_parameters,
)
from sklearn.preprocessing import OrdinalEncoder
from transformers import (
    EarlyStoppingCallback, Trainer, TrainingArguments  # type: ignore
)
from safetensors.torch import load_file


logger = logging.getLogger(__name__)


def df_split_time_aware(
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

    def to_dict(self) -> Dict[str, Any]:
        output = super(TimeSeriesPreprocessor, self).to_dict()

        for k, v in self.scaler_dict.items():
            output["scaler_dict"][k] = v.to_dict()

        for k, v in self.target_scaler_dict.items():
            output["target_scaler_dict"][k] = v.to_dict()

        if self.scaling_id_columns and self.scaling:
            akey = next(iter(self.target_scaler_dict.keys()))
            if isinstance(akey, tuple):
                key_types = [type(k) for k in akey]
            else:
                key_types = [type(akey)]
        else:
            key_types = []

        output["scaling_id_columns_types"] = [
            TYPE_TO_STRING[k] for k in key_types
        ]

        # PATCH: manual serialization of OrdinalEncoder
        if self.categorical_encoder and isinstance(
            self.categorical_encoder, OrdinalEncoder
        ):
            encoder = self.categorical_encoder
            output["categorical_encoder"] = {
                "type": "sklearn.preprocessing.OrdinalEncoder",
                "categories": [list(c) for c in encoder.categories_],  # type: ignore
                "handle_unknown": encoder.handle_unknown,  # type: ignore
                "unknown_value": encoder.unknown_value,  # type: ignore
            }
        elif self.categorical_encoder:
            raise ValueError("Unsupported encoder type for serialization.")

        return output

    @classmethod
    def from_dict(
        cls,
        feature_extractor_dict: Dict[str, Any],
        **kwargs
    ) -> "SafeTimeSeriesPreprocessorOrdinal":
        obj = super().from_dict(feature_extractor_dict, **kwargs)

        cat_enc = feature_extractor_dict.get("categorical_encoder")
        logger.info(cat_enc)
        if (
            isinstance(cat_enc, dict)
            and cat_enc.get("type") == "sklearn.preprocessing.OrdinalEncoder"
        ):
            encoder = OrdinalEncoder(
                handle_unknown=cat_enc.get("handle_unknown", "error"),
                unknown_value=cat_enc.get("unknown_value", None),
            )
            encoder.categories_ = [np.array(c) for c in cat_enc["categories"]]
            obj.categorical_encoder = encoder

        return obj


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

    logger.info(f"📥 Chargement TimeSeries Preprocessor depuis {load_dir}")

    return tsp


def save_granite_model(experiment, model_results: dict, save_dir: str = "."):
    exp_params = model_results["exp_params"]
    best_checkpoint = model_results["best_checkpoint"]
    name = exp_params["name"].replace(" ", "-")
    ctxt = f"c{exp_params['context_length']}"
    fcm_ctxt = f"fcm{exp_params['fcm_context_length']}"
    pred = f"p{exp_params['prediction_length']}"
    start = exp_params["sub_range"][0]
    sub_range = exp_params["sub_range"]
    stop = sub_range[1] - 1 if len(sub_range) > 1 else "end"

    filename = (
        f"granite_results_{name}"
        f"_{best_checkpoint}_{ctxt}_{fcm_ctxt}_{pred}_{start}-{stop}.joblib"
    )
    filepath = f"{save_dir}/{filename}"
    joblib.dump(model_results, filepath)

    logger.info(f"{experiment} saved in {filepath}")


def train_or_resume(
    trainer: Trainer,
    exp_params: dict = {},
) -> str:
    name = exp_params["name"]
    out_dir = exp_params["out_dir"]
    logger.info("\n" + pprint.pformat(exp_params))
    if "best_checkpoint" in exp_params:
        best_checkpoint = exp_params['best_checkpoint']
        checkpoint_dir = os.path.join(out_dir, f"{name}_output", best_checkpoint)
    else:
        checkpoint_dir = ""
    logger.info(f"Checkpoint_dir identified : [{checkpoint_dir}]")
    if os.path.isdir(checkpoint_dir):
        model_weights_path = os.path.join(checkpoint_dir, "model.safetensors")
        logger.info(f"🔁 Loading weights only from {model_weights_path}")
        state_dict = load_file(model_weights_path)
        trainer.model.load_state_dict(state_dict)  # type: ignore
        trainer.train()  # ⚠ repart from scratch mais avec les bons poids
    else:
        logger.info("🆕 Starting training from scratch")
        trainer.train()
    logger.info(f"Best checkpoint: {trainer.state.best_model_checkpoint}")
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

    logger.info(f"📥 Chargement config du modèle depuis {config_path}")
    model = model_class.from_pretrained(checkpoint_dir, local_files_only=True)

    logger.info(f"🔁 Chargement poids du modèle uniquement depuis {weights_path}")
    state_dict = load_file(weights_path, device=device)
    model.load_state_dict(state_dict)

    return model


def summarize_module_structure(model, use_netron=False):
    """
    Print model structure and optionally launch Netron if available.

    Args:
        model (nn.Module): Your PyTorch model (e.g. TinyTimeMixer).
    """
    if use_netron:  # type:ignore
        model_name = model.__class__.__name__
        outdir = os.path.join("./ttm_torchsave.model", model_name)
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, "model.pt")
        torch.save(model.state_dict(), path)
        netron.start(path)  # type:ignore
    else:
        print(model)


def fine_tune_model(
    output_dir,
    logging_dir,
    context_length,
    prediction_length,
    fcm_context_length,
    column_specifiers,
    learning_rate,
    num_epochs,
    batch_size,
    steps_per_epoch,
    patience,
    device,
):
    # Define preprocessor
    finetune_tsp = SafeTimeSeriesPreprocessorOrdinal(
        **column_specifiers,
        context_length=context_length,
        prediction_length=prediction_length,
        scaling=True,
        freq="h",
        encode_categorical=True,
        scale_categorical_columns=True,
        scaler_type="standard",  # type: ignore
    )

    # Define model from Hugging Face
    finetune_forecast_model = TinyTimeMixerForPrediction.from_pretrained(
        "ibm-granite/granite-timeseries-ttm-r2",  # Name of the model on HuggingFace.
        num_input_channels=finetune_tsp.num_input_channels,
        prediction_channel_indices=finetune_tsp.prediction_channel_indices,
        exogenous_channel_indices=finetune_tsp.exogenous_channel_indices,
        fcm_use_mixer=True,
        fcm_context_length=fcm_context_length,
        enable_forecast_channel_mixing=True,
        fcm_mix_layers=2,
        decoder_mode="mix_channel",
        fcm_prepend_past=True,
    )
    logger.info(f"Bascule du modèle sur {device}")
    finetune_forecast_model.to(device)  # type: ignore
    summarize_module_structure(finetune_forecast_model)

    # Freeze the backbone of the model
    logger.info(f"Number of params before freezing backbone {
        count_parameters(finetune_forecast_model)
    }")
    for param in finetune_forecast_model.backbone.parameters():
        param.requires_grad = False
    logger.info(f"Number of params after freezing the backbone {
        count_parameters(finetune_forecast_model)
    }")

    # Set the training arguments
    logger.info(f"Learning Rate = {learning_rate} | {num_epochs} epoch(s) |"
                f" utilisation du {'GPU' if (device == 'cuda') else 'CPU'}")
    finetune_forecast_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        do_eval=True,
        eval_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        dataloader_pin_memory=True,
        report_to=None,
        save_strategy="epoch",
        logging_strategy="epoch",
        save_total_limit=1,
        logging_dir=logging_dir,  # Make sure to specify a logging directory
        load_best_model_at_end=True,  # Load the best model when training ends
        metric_for_best_model="eval_loss",  # Metric to monitor for early stopping
        greater_is_better=False,  # For loss
        use_cpu=device != "cuda",
    )

    # Create the early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        # Number of epochs with no improvement after which to stop
        early_stopping_patience=patience,
        # Minimum improvement required to consider as improvement
        early_stopping_threshold=0.00001,
    )
    tracking_callback = TrackingCallback()

    # Define an optimizer and scheduler
    finetune_optimizer = AdamW(finetune_forecast_model.parameters(), lr=learning_rate)
    finetune_scheduler = OneCycleLR(
        finetune_optimizer,
        learning_rate,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
    )
    return (
        finetune_tsp,
        finetune_forecast_model,
        finetune_forecast_args,
        finetune_optimizer, finetune_scheduler,
        early_stopping_callback,
        tracking_callback
    )
