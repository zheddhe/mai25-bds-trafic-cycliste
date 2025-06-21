import logging
import logging.config
import sys


def setup_logger(level: int = logging.INFO) -> None:
    """
    Configure logging globally with a stream handler.

    Args:
        level (int): Logging level, e.g., logging.DEBUG or logging.INFO.
    """
    LOGGING_CONFIG = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '[%(levelname)s]-[%(asctime)s] [%(name)s] %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': sys.stdout
            },
        },
        'root': {
            'handlers': ['console'],
            'level': level
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)
