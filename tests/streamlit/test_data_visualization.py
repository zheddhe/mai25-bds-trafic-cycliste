import pandas as pd
import os
import pytest
from typing import cast
import app.pages.data_visualization as dv


@pytest.fixture(autouse=True)
def enable_test_mode():
    os.environ["IS_TESTING"] = "1"
    dv.cached_load_dataset_exploration.clear()  # type: ignore
    yield
    del os.environ["IS_TESTING"]


def test_dataset_structure_default():
    df = dv.cached_load_dataset_exploration()
    df = cast(pd.DataFrame, df)
    assert df.shape == (1, 7)
    assert "comptage_horaire" in df.columns


def test_add_column_and_run_boxplot():
    df = dv.cached_load_dataset_exploration()
    df = cast(pd.DataFrame, df)
    df["mois_annee_comptage"] = ["2023-01"]
    fig = dv.px.box(df, x="mois_annee_comptage", y="comptage_horaire")
    assert fig.data[0].type == "box"  # type: ignore


def test_add_dayname_and_run_barplot():
    df = dv.cached_load_dataset_exploration()
    df = cast(pd.DataFrame, df)
    df["date_et_heure_de_comptage_dayname"] = ["Monday"]
    grouped = df.groupby("date_et_heure_de_comptage_dayname")[
        "comptage_horaire"].mean().reset_index()
    fig = dv.px.bar(grouped, x="date_et_heure_de_comptage_dayname",
                    y="comptage_horaire")
    assert fig.data[0].type == "bar"  # type: ignore


def test_add_hour_and_run_top_hours():
    df = dv.cached_load_dataset_exploration()
    df = cast(pd.DataFrame, df)
    df["date_et_heure_de_comptage_hour"] = [8]
    df2 = df.groupby("date_et_heure_de_comptage_hour")[
        "comptage_horaire"].sum().reset_index()
    fig = dv.px.bar(df2, x="date_et_heure_de_comptage_hour",
                    y="comptage_horaire")
    assert fig.data[0].type == "bar"  # type: ignore
