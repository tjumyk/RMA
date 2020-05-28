import logging_config
from const_vars.const_vars_abc import Dictionaries

log = logging_config.get_logger()


class Dict_Impl(Dictionaries):
    dataset_type = 'test'
    dataset_source = 'ace2014_uiuc'
