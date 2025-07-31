from pathlib import Path
from unittest.mock import patch
import importlib
import app.sections.project_presentation as project_presentation
from streamlit.delta_generator import DeltaGenerator


def test_warning_called_for_missing_images():
    # Importing a page triggers its execution
    with patch.object(Path, "exists", return_value=False), \
         patch.object(DeltaGenerator, "warning") as mock_warning:
        importlib.reload(project_presentation)

        # number of warning blocks
        expected_calls = 47

        assert mock_warning.call_count == expected_calls
