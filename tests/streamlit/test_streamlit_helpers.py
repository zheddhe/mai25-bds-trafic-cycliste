from app.utils.streamlit_helpers import example_helper


def test_example_helper_returns_string():
    assert example_helper() == "This is a helper function"
