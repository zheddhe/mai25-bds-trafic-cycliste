from app import config


def test_app_name_is_defined():
    assert isinstance(config.APP_NAME, str)
    assert config.APP_NAME == "Application Trafic Cycliste"
