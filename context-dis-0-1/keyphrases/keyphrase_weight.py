#!/usr/bin/env python
import os
import math
from const_vars.const_vars_abc import Dictionaries
from nltk.stem import WordNetLemmatizer
import decimal

# data_dir = os.path.dirname(nerd.__path__[0])


def get_token_weight_idf(token):
    # todo need to Lemmatizer token
    if token not in Dictionaries.word_idf_dict:
        weight = math.log2(int(Dictionaries.collection_size))
    else:
        weight = Dictionaries.word_idf_dict[token]
    return weight


def get_token_weight_mi(entity, token):
    # todo need to Lemmatizer token and update the weight
    if (entity, token) not in Dictionaries.word_mi_dict:
        weight = 0.001  # todo update it.
    else:
        weight = Dictionaries.word_mi_dict[(entity, token)]
    return weight


def get_token_weight_aida(token):
    weight = math.log2(int(Dictionaries.collection_size) / (Dictionaries.word_count_dict.get(token, 0) + 1))
    return weight


def document_frequency():
    pass


if __name__ == '__main__':
    nerd.constant.init_dictionaries()
    print(get_token_weight_mi('Stephen_L._Gunn','community'))
