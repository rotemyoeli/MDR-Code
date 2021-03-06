import contextlib
import sys
import pathlib
import logging

def init_logging(log_file_name):
    log_path = pathlib.Path(log_file_name)
    log_path.parent.mkdir(parents=True,exist_ok=True)
    logging.config.dictConfig({
        'version': 1,
        'formatters': {
            'default': {'format': '%(asctime)s [%(name)-10s] %(levelname)-5s\t%(message)s',
                        'datefmt': '%Y-%m-%d_%H-%M-%S'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': log_file_name,
                'maxBytes': 1000 * 1024 * 1024,
                'backupCount': 3
            },
            'file': {
                'level': 'INFO',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': log_path.parent / (log_path.stem + '-INFO' + log_path.suffix),
                'maxBytes': 10 * 1024 * 1024,
                'backupCount': 3
            }
        },
        'loggers': {
            '': {
                'level': 'DEBUG',
                'handlers': ['console', 'file'],
                'propagate': True
            }
        },
        'disable_existing_loggers': False
    })
    logging.getLogger('parso').setLevel(logging.ERROR)
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('main').info(f'=== STARTED: {sys.argv}')


@contextlib.contextmanager
def log_context(description, loglevel=logging.DEBUG):
    logging.getLogger(__name__).log(loglevel, '>>>> STARTED %s' % description)
    yield None
    logging.getLogger(__name__).log(loglevel, '<<<< FINISHED %s' % description)