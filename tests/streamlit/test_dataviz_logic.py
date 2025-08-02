import pandas as pd
from io import StringIO
from typing import cast
from unittest.mock import patch, Mock
from app.utils.dataviz_logic import (
    cached_load_dataset_visualization,
    display_distrib_counting_graphics,
    manage_dataset_visualization,
    display_average_counting_graphics,
    display_total_counting_graphics,
    display_distrib_multi_level_graphics,
)


def test_cached_load_dataset_visualization_uploaded_file(monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    cached_load_dataset_visualization.clear()  # type: ignore
    csv_data = "index,col1\n0,foo\n1,bar"
    uploaded_file = StringIO(csv_data)

    df = cached_load_dataset_visualization(uploaded_file)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["col1"]
    assert df.shape == (2, 1)


@patch("app.utils.dataviz_logic.load_dataset_from_config")
def test_cached_load_dataset_visualization_from_config(mock_loader, monkeypatch):
    monkeypatch.delenv("IS_TESTING", raising=False)
    cached_load_dataset_visualization.clear()  # type: ignore
    mock_df = pd.DataFrame({"x": [1]})
    mock_loader.return_value = mock_df

    df = cached_load_dataset_visualization(None)
    df = cast(pd.DataFrame, df)

    mock_loader.assert_called_once()
    assert df.equals(mock_df)


def test_display_distrib_counting_graphics_dayname(monkeypatch):
    df = pd.DataFrame({
        "date_et_heure_de_comptage_dayname": ["Monday", "Tuesday", "Monday"],
        "comptage_horaire": [10, 20, 30]
    })
    mock_st = Mock()
    col1, col2 = Mock(), Mock()
    mock_st.columns.return_value = (col1, col2)
    col2.selectbox.return_value = "date_et_heure_de_comptage_dayname"

    display_distrib_counting_graphics(df, st_module=mock_st)

    mock_st.plotly_chart.assert_called_once()


def test_display_distrib_counting_graphics_hour(monkeypatch):
    df = pd.DataFrame({
        "date_et_heure_de_comptage_hour": [8, 9, 8],
        "comptage_horaire": [15, 25, 35]
    })
    mock_st = Mock()
    col1, col2 = Mock(), Mock()
    mock_st.columns.return_value = (col1, col2)
    col2.selectbox.return_value = "date_et_heure_de_comptage_hour"

    display_distrib_counting_graphics(df, st_module=mock_st)
    called_args, _ = mock_st.plotly_chart.call_args
    fig = called_args[0]
    x_values = [trace.x for trace in fig.data if hasattr(trace, "x")]

    assert any("8h" in x or "9h" in x for x in x_values[0])


def test_manage_dataset_reload_triggers_clear_and_rerun(monkeypatch):
    monkeypatch.setenv("IS_TESTING", "1")
    cached_load_dataset_visualization.clear()  # type: ignore
    mock_st = Mock()
    mock_st.button.return_value = True  # simulate click
    mock_st.spinner.return_value.__enter__ = lambda s: None
    mock_st.spinner.return_value.__exit__ = lambda s, exc, val, tb: None
    mock_st.rerun = Mock()

    with patch("app.utils.dataviz_logic."
               "cached_load_dataset_visualization.clear") as mock_clear:
        manage_dataset_visualization(mock_st)

    mock_clear.assert_called_once()
    mock_st.rerun.assert_called_once()


def test_manage_dataset_triggers_error_and_empty_df(monkeypatch):
    monkeypatch.setenv("IS_TESTING", "1")
    cached_load_dataset_visualization.clear()  # type: ignore
    mock_st = Mock()
    mock_st.button.return_value = False
    mock_st.file_uploader.return_value = None
    mock_st.spinner.return_value.__enter__ = lambda s: None
    mock_st.spinner.return_value.__exit__ = lambda s, exc, val, tb: None
    mock_st.stop = Mock()

    with patch("app.utils.dataviz_logic.cached_load_dataset_visualization",
               return_value="not_a_dataframe"):
        df_result = manage_dataset_visualization(mock_st)

    mock_st.error.assert_called_once()
    mock_st.stop.assert_called_once()
    assert isinstance(df_result, pd.DataFrame)
    assert df_result.empty


def test_display_average_counting_graphics_monthname(monkeypatch):
    df = pd.DataFrame({
        "date_et_heure_de_comptage_monthname": ["January", "February", "March"],
        "comptage_horaire": [10, 20, 30]
    })
    mock_st = Mock()
    col1, col2 = Mock(), Mock()
    mock_st.columns.return_value = (col1, col2)
    col2.selectbox.return_value = "date_et_heure_de_comptage_monthname"

    display_average_counting_graphics(df, st_module=mock_st)

    mock_st.plotly_chart.assert_called_once()


def test_display_total_counting_graphics_dayname(monkeypatch):
    df = pd.DataFrame({
        "date_et_heure_de_comptage_dayname": ["Monday", "Tuesday", "Monday"],
        "comptage_horaire": [10, 20, 30]
    })
    mock_st = Mock()
    col1, col2 = Mock(), Mock()
    mock_st.columns.return_value = (col1, col2)
    col2.selectbox.return_value = "date_et_heure_de_comptage_dayname"

    display_total_counting_graphics(df, st_module=mock_st)

    mock_st.plotly_chart.assert_called_once()


def test_display_distrib_multi_level_graphics_same_columns_triggers_warning():
    df = pd.DataFrame({
        "arrondissement": ["01", "02", "03"],
        "comptage_horaire": [100, 200, 300],
        "date_et_heure_de_comptage_dayname": ["Monday", "Tuesday", "Wednesday"],
        "date_et_heure_de_comptage_hour": [8, 9, 10],
        "nom_du_site_de_comptage": ["A", "B", "C"]
    })
    mock_st = Mock()
    col_title, col1, col2 = Mock(), Mock(), Mock()
    mock_st.columns.return_value = (col_title, col1, col2)
    col1.selectbox.return_value = "date_et_heure_de_comptage_dayname"
    col2.selectbox.return_value = "date_et_heure_de_comptage_dayname"

    display_distrib_multi_level_graphics(df, st_module=mock_st)

    mock_st.warning.assert_called_once_with(
        "Les niveaux 1 et 2 doivent être différents pour Sunburst."
    )
    mock_st.plotly_chart.assert_not_called()
