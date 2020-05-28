""" Logging Configuration """
from __future__ import absolute_import
import logging
# import coloredlogs


def setup():
    fmt = '%(asctime)s|%(levelname)s|%(module)s|%(lineno)d|%(message)s'
    formatter = logging.Formatter(fmt)

    logger = logging.getLogger('nerd')
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    # fh = logging.FileHandler('app.log')
    # fh.setLevel(logging.INFO)
    # fh.setFormatter(fmt)
    # logging.basicConfig(format=fmt)

    logger.addHandler(ch)
    # coloredlogs.install(level="INFO", fmt=fmt)


def get_logger():
    return logging.getLogger('nerd')


setup()
