import logging_config
from const_vars.const_vars_abc import Dictionaries

log = logging_config.get_logger()


class Dict_Co(Dictionaries):
    dataset_type = 'testb'

    @staticmethod
    def set_dataset_type(dataset_type):
        Dict_Co.dataset_type = dataset_type
