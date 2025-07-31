import pandas as pd
import pytest
import os
from typing import cast
import app.sections.data_exploration as de


@pytest.fixture(autouse=True)
def enable_test_mode():
    os.environ["IS_TESTING"] = "1"
    de.cached_load_dataset_exploration.clear()  # type: ignore
    yield
    del os.environ["IS_TESTING"]


def test_dataset_exploration_structure():
    df = de.cached_load_dataset_exploration(None)
    df = cast(pd.DataFrame, df)
    assert df.shape == (2, 4)
    assert sorted(df.columns) == sorted([
        "nom_du_site_de_comptage",
        "orientation_compteur",
        "comptage_horaire",
        "date_et_heure_de_comptage"
    ])
