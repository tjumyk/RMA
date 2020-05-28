#!/usr/bin/env python
import os
import math
from const_vars.const_vars_abc import Dictionaries
import decimal

# data_dir = os.path.dirname(nerd.__path__[0])


def get_token_weight_idf(token, redis_key='idf'):
    weight = Dictionaries.redis_db_obj.get_word_weight_redis(redis_key, token)
    return weight if weight else math.log2(int(Dictionaries.collection_size))


def get_token_weight_mi(entity, token, redis_key='mi'):
    token = str((entity, token))
    weight = Dictionaries.redis_db_obj.get_word_weight_redis(redis_key, token)
    # todo smoothing score for pointwise mutual information
    return weight if weight else 0.00001


def get_token_weight_aida(token):
    weight = math.log2(int(Dictionaries.collection_size) / (Dictionaries.word_count_dict.get(token, 0) + 1))
    return weight


def document_frequency():
    pass


if __name__ == '__main__':
    nerd.constant.init_dictionaries()
    print(get_token_weight_mi('Stephen_L._Gunn','community'))
