import database
import hashlib
import os
import database.db_kongzi2
import database.db_kongzi3
from collections import defaultdict
import traceback
import re
import logging_config
import spacy
import csv
import requests
from bs4 import BeautifulSoup
from const_vars.const_vars_abc import Dictionaries

log = logging_config.get_logger()


class Dict_Co(Dictionaries):
    dataset_type = 'testb'
    res_path = 'testb'

    @staticmethod
    def set_dataset_type(dataset_type):
        Dict_Co.dataset_type = dataset_type
