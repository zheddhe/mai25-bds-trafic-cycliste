import pandas as pd
import json
import os
from unittest.mock import patch, MagicMock, mock_open
from smartcheck.deep_learning_project_specific import (
    df_split_time_aware,
    SafeTimeSeriesPreprocessorOrdinal,
    save_preprocessor_state,
    load_preprocessor_state,
    train_or_resume,
    load_model_from_checkpoint,
    summarize_module_structure,
    fine_tune_model,
    save_granite_model,
)
from sklearn.preprocessing import OrdinalEncoder


class TestDfSplitTimeAware:
    """Unit tests for df_split_time_aware"""

    # === Tests ===
    def test_split_respects_order_and_ratio(self):
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="D"),
            "value": range(10)
        })
        train, test = df_split_time_aware(df, "timestamp", test_size=0.2)
        assert len(train) == 8
        assert len(test) == 2
        assert train["timestamp"].max() < test["timestamp"].min()


class TestSafeTimeSeriesPreprocessorOrdinal:
    """Unit tests for SafeTimeSeriesPreprocessorOrdinal"""

    # === Tests ===
    def test_unknown_handling_and_encoding(self):
        df_train = pd.DataFrame({"id": [1, 2, 3], "cat": ["a", "b", "c"]})
        df_test = pd.DataFrame({"id": [4], "cat": ["zzz"]})

        tsp = SafeTimeSeriesPreprocessorOrdinal(
            timestamp_column="id",
            id_columns=["id"],
            target_columns=["cat"],
            categorical_columns=["cat"],
            static_categorical_columns=[],
            observable_columns=[],
            control_columns=[],
            conditional_columns=[],
            context_length=2,
            prediction_length=1,
            scaling=False,
            encode_categorical=True,
            freq="h"
        )

        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=999
        )
        encoder.fit(df_train[["cat"]])
        tsp.categorical_encoder = encoder

        encoded = tsp._process_encoding(df_test)
        assert encoded["cat"].iloc[0] == 999.0

    def test_to_dict_structure(self):
        df = pd.DataFrame({"id": [1], "cat": ["a"]})
        tsp = SafeTimeSeriesPreprocessorOrdinal(
            timestamp_column="id",
            id_columns=["id"],
            target_columns=["cat"],
            categorical_columns=["cat"],
            static_categorical_columns=[],
            observable_columns=[],
            control_columns=[],
            conditional_columns=[],
            context_length=1,
            prediction_length=1,
            scaling=True,
            encode_categorical=True,
            freq="h"
        )
        tsp._train_categorical_encoder(
            tsp._replace_unknowns(df, ["cat"], force_include=True)
        )
        mock_scaler = MagicMock()
        mock_scaler.to_dict.return_value = {}
        key = json.dumps(["cat"])
        tsp.scaler_dict = {key: mock_scaler}
        tsp.target_scaler_dict = {key: mock_scaler}
        tsp_dict = tsp.to_dict()
        assert "categorical_encoder" in tsp_dict
        assert tsp_dict[
            "categorical_encoder"
        ]["type"] == "sklearn.preprocessing.OrdinalEncoder"

    @patch("smartcheck.deep_learning_project_specific.TimeSeriesPreprocessor.from_dict")
    def test_from_dict_with_valid_encoder(self, mock_super_from_dict):
        # --- Mock du super ---
        mock_obj = MagicMock()
        mock_super_from_dict.return_value = mock_obj

        # --- Dict simulant l'état sauvegardé ---
        input_dict = {
            "categorical_encoder": {
                "type": "sklearn.preprocessing.OrdinalEncoder",
                "categories": [["a", "b"], ["x", "y"]],
                "handle_unknown": "use_encoded_value",
                "unknown_value": 999,
            }
        }

        result = SafeTimeSeriesPreprocessorOrdinal.from_dict(input_dict)

        mock_super_from_dict.assert_called_once_with(input_dict)
        enc_cat = result.categorical_encoder
        assert isinstance(enc_cat, OrdinalEncoder)
        assert enc_cat.handle_unknown == "use_encoded_value"  # type: ignore
        assert enc_cat.unknown_value == 999  # type: ignore
        assert enc_cat.categories_[0].tolist() == ["a", "b"]  # type: ignore


class TestSavePreprocessorState:
    """Unit tests for save_preprocessor_state"""

    # === Tests ===
    @patch("smartcheck.deep_learning_project_specific.yaml.safe_dump")
    @patch("smartcheck.deep_learning_project_specific.joblib.dump")
    @patch("smartcheck.deep_learning_project_specific.os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    def test_save_preprocessor_state(
        self, mock_open_f, makedirs, dump, dump_yaml
    ):
        tsp = MagicMock(spec=SafeTimeSeriesPreprocessorOrdinal)
        tsp.timestamp_column = "ts"
        tsp.id_columns = ["id"]
        tsp.target_columns = ["target"]
        tsp.categorical_columns = ["cat"]
        tsp.static_categorical_columns = []
        tsp.observable_columns = []
        tsp.control_columns = []
        tsp.conditional_columns = []
        tsp.context_length = 1
        tsp.prediction_length = 1
        tsp.scaling = True
        tsp.scaler_type = "standard"
        tsp.encode_categorical = True
        tsp.freq = "h"
        tsp.categorical_encoder = MagicMock()
        tsp.scaler_dict = {}
        tsp.target_scaler_dict = {}

        save_preprocessor_state(tsp, "tmp")
        assert dump.call_count == 3

    @patch("smartcheck.deep_learning_project_specific.yaml.safe_load")
    @patch("smartcheck.deep_learning_project_specific.joblib.load")
    @patch("builtins.open", new_callable=mock_open, read_data="{}")
    def test_load_preprocessor_state(self, mock_open_f, joblib_load, yaml_load):
        yaml_load.return_value = {
            "timestamp_column": "ts",
            "id_columns": ["id"],
            "target_columns": ["target"],
            "categorical_columns": ["cat"],
            "static_categorical_columns": [],
            "observable_columns": [],
            "control_columns": [],
            "conditional_columns": [],
            "context_length": 1,
            "prediction_length": 1,
            "scaling": True,
            "scaler_type": "standard",
            "encode_categorical": True,
            "freq": "h",
        }
        tsp = load_preprocessor_state("tmp")
        assert isinstance(tsp, SafeTimeSeriesPreprocessorOrdinal)


class TestTrainOrResume:
    """Unit tests for train_or_resume"""

    @patch("smartcheck.deep_learning_project_specific.load_file")
    def test_train_or_resume_no_checkpoint(self, load_file):
        trainer = MagicMock()
        trainer.state.best_model_checkpoint = "checkpoint-123"
        exp_params = {"name": "exp1", "out_dir": os.path.join(os.sep, "tmp")}

        with patch("os.path.isdir", return_value=False):
            result = train_or_resume(trainer, exp_params)

        assert result == "checkpoint-123"
        trainer.train.assert_called_once()
        load_file.assert_not_called()

    @patch("smartcheck.deep_learning_project_specific.os.path.isdir")
    @patch("smartcheck.deep_learning_project_specific.load_file")
    def test_resume_training_from_checkpoint(
        self, mock_load_file, mock_isdir
    ):
        trainer = MagicMock()
        trainer.state.best_model_checkpoint = "checkpoint-42"
        trainer.model.load_state_dict.return_value = None
        mock_isdir.return_value = True

        out_dir = os.path.join(os.sep, "tmp")
        exp_params = {
            "name": "testexp",
            "out_dir": out_dir,
            "best_checkpoint": "checkpoint-42"
        }

        checkpoint_dir = os.path.join(out_dir, "testexp_output", "checkpoint-42")
        expected_weights_path = os.path.join(checkpoint_dir, "model.safetensors")

        with patch("smartcheck.deep_learning_project_specific.logging.info"):
            result = train_or_resume(trainer, exp_params)

        assert result == "checkpoint-42"
        trainer.train.assert_called_once()
        trainer.model.load_state_dict.assert_called_once()
        assert os.path.normpath(mock_load_file.call_args[0][0]) == \
            os.path.normpath(expected_weights_path)


class TestLoadModelFromCheckpoint:
    """Unit tests for load_model_from_checkpoint"""

    # === Tests ===
    @patch("smartcheck.deep_learning_project_specific.load_file")
    def test_load_model_from_checkpoint(self, load_file):
        dummy_model = MagicMock()
        dummy_model_class = MagicMock()
        dummy_model_class.from_pretrained.return_value = dummy_model

        with patch("os.path.isfile", return_value=True):
            model = load_model_from_checkpoint(
                dummy_model_class, "some_dir", device="cpu"
            )
        assert model == dummy_model


class TestSummarizeModuleStructure:
    """Unit tests for summarize_module_structure"""

    # === Tests ===
    @patch("smartcheck.deep_learning_project_specific.netron.start")
    @patch("smartcheck.deep_learning_project_specific.torch.save")
    @patch("smartcheck.deep_learning_project_specific.os.makedirs")
    def test_summarize_with_netron(self, makedirs, torch_save, netron_start):
        model = MagicMock()
        model.__class__.__name__ = "DummyModel"
        summarize_module_structure(model, use_netron=True)
        torch_save.assert_called_once()
        netron_start.assert_called_once()


class TestFineTuneModel:
    """Unit tests for fine_tune_model"""

    # === Tests ===
    @patch(
        "smartcheck.deep_learning_project_specific."
        "TinyTimeMixerForPrediction.from_pretrained"
    )
    @patch("smartcheck.deep_learning_project_specific.AdamW")
    @patch("smartcheck.deep_learning_project_specific.OneCycleLR")
    def test_fine_tune_model_returns_expected(
        self, mock_scheduler, mock_adamw, mock_model_cls
    ):
        model = MagicMock()
        mock_model_cls.return_value = model

        out = fine_tune_model(
            output_dir="/tmp",
            logging_dir="/tmp/logs",
            context_length=2,
            prediction_length=1,
            fcm_context_length=2,
            column_specifiers={
                "timestamp_column": "ts",
                "id_columns": ["id"],
                "target_columns": ["y"],
                "categorical_columns": ["cat"],
                "static_categorical_columns": [],
                "observable_columns": [],
                "control_columns": [],
                "conditional_columns": []
            },
            learning_rate=1e-3,
            num_epochs=1,
            batch_size=4,
            steps_per_epoch=1,
            patience=2,
            device="cpu"
        )
        assert len(out) == 7


class TestSaveGraniteModel:
    """Unit tests for save_granite_model"""

    # === Tests ===
    @patch("smartcheck.deep_learning_project_specific.joblib.dump")
    def test_save_granite_model_generates_expected_filename(self, mock_dump):
        model_results = {
            "exp_params": {
                "name": "Granite Demo",
                "context_length": 24,
                "fcm_context_length": 168,
                "prediction_length": 12,
                "sub_range": [0, 300],
            },
            "best_checkpoint": "checkpoint-88"
        }
        experiment = "Experiment01"

        with patch(
            "smartcheck.deep_learning_project_specific.logger.info"
        ) as log_info:
            save_granite_model(experiment, model_results, save_dir="output")

        # Recompose expected path (no regex)
        expected_filename = (
            "granite_results_Granite-Demo_checkpoint-88_"
            "c24_fcm168_p12_0-299.joblib"
        )
        expected_path = os.path.normpath(os.path.join("output", expected_filename))
        actual_path_arg = os.path.normpath(mock_dump.call_args[0][1])

        assert mock_dump.called
        assert actual_path_arg == expected_path
        log_info.assert_called_once()
        assert experiment in log_info.call_args[0][0]
