import spacy
import logging_config
import sys
import argparse
import time
import json
import os
import database.db_kongzi3
import const_vars.constant_conll_testb
import corpora.conll
import result_folder
from collections import defaultdict
from tokenizer import spacy_tokenizer
import database.db_kongzi2
import redis_db.redis_init
# from feature_generator.fea_gen_ctx_dis_1 import process
# from feature_generator.fea_gen_ctx_basic import process


log = logging_config.get_logger()

config_path = database.__path__[0]
config_path = os.path.join(config_path, 'config.json')
result_path = result_folder.__path__[0]

with open(config_path, 'r') as f:
    config = json.load(f)
threads_num = config['threads_num']


log.info("Threads for processing the data: {}".format(threads_num))
const_vars_dict = const_vars.constant_conll_testb.Dict_Co
const_vars_dict.init_dictionaries()


const_vars_dict.threads = threads_num


def phrase_extraction(q_id, w_size=50, c_size=30):
    s_t = time.time()
    log.info("Generating entity linking queries...")
    cases_list, men, m_text = corpora.conll.get_valid_mention_entity_pairs(q_id)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    ent_art_dict = defaultdict(str)
    # spacy_tokenizer.main_gen_web(cases_list, men, m_text)
    spacy_tokenizer.main_gen(cases_list, men, m_text)


def phrase_extraction_web_co_redis(q_id, w_size=50, c_size=30):
    s_t = time.time()
    const_vars_dict.db_conn_obj = database.db_kongzi3.PostgreSQL()
    const_vars_dict.prior_db_conn_obj = database.db_kongzi2.PostgreSQLAIDA()
    const_vars_dict.redis_db_obj = redis_db.redis_init.RedisConn()
    log.info("Generating entity linking queries...")
    cases_list, men, m_text, doc_id, g_ent = corpora.conll.\
        get_valid_mention_entity_pairs(q_id)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    # ent_art_dict = defaultdict(str)
    result = spacy_tokenizer.main_gen_web_co_redis(cases_list, men, m_text, doc_id)
    result['gold'] = g_ent
    result['men'] = men
    return result


def men_ents_phrase_web_co_redis(q_id, w_size=50, c_size=30):
    s_t = time.time()
    const_vars_dict.db_conn_obj = database.db_kongzi3.PostgreSQL()
    const_vars_dict.prior_db_conn_obj = database.db_kongzi2.PostgreSQLAIDA()
    const_vars_dict.redis_db_obj = redis_db.redis_init.RedisConn()
    log.info("Generating entity linking queries...")
    men = corpora.conll.get_mention_by_qid(q_id)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    # ent_art_dict = defaultdict(str)
    result = spacy_tokenizer.main_gen_web_co_redis(men)
    result['gold'] = g_ent
    result['men'] = men
    return result


def phrase_extraction_web_fq_redis(q_id, w_size=50, c_size=30):
    s_t = time.time()
    const_vars_dict.db_conn_obj = database.db_kongzi3.PostgreSQL()
    const_vars_dict.prior_db_conn_obj = database.db_kongzi2.PostgreSQLAIDA()
    const_vars_dict.redis_db_obj = redis_db.redis_init.RedisConn()
    log.info("Generating entity linking queries...")
    cases_list, men, m_text, doc_id, g_ent = corpora.conll.get_valid_mention_entity_pairs(q_id)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    # ent_art_dict = defaultdict(str)
    result = spacy_tokenizer.main_gen_web_fq_redis(cases_list, men, m_text, doc_id)
    result['gold'] = g_ent
    result['men'] = men
    return result


def phrase_extraction_web_idf_redis(q_id, w_size=50, c_size=30):
    s_t = time.time()
    const_vars_dict.db_conn_obj = database.db_kongzi3.PostgreSQL()
    const_vars_dict.prior_db_conn_obj = database.db_kongzi2.PostgreSQLAIDA()
    const_vars_dict.redis_db_obj = redis_db.redis_init.RedisConn()
    log.info("Generating entity linking queries...")
    cases_list, men, m_text, doc_id, g_ent = corpora.conll.get_valid_mention_entity_pairs(q_id)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    # ent_art_dict = defaultdict(str)
    result = spacy_tokenizer.main_gen_web_idf_redis(cases_list, men, m_text, doc_id)
    result['gold'] = g_ent
    result['men'] = men
    return result


def phrase_extraction_web_idf_entropy_redis(q_id, w_size=50, c_size=30):
    s_t = time.time()
    const_vars_dict.db_conn_obj = database.db_kongzi3.PostgreSQL()
    const_vars_dict.prior_db_conn_obj = database.db_kongzi2.PostgreSQLAIDA()
    const_vars_dict.redis_db_obj = redis_db.redis_init.RedisConn()
    log.info("Generating entity linking queries...")
    cases_list, men, m_text, doc_id, g_ent = corpora.conll.get_valid_mention_entity_pairs(q_id)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    # ent_art_dict = defaultdict(str)
    result = spacy_tokenizer.main_gen_web_idf_entropy_redis(cases_list, men, m_text, doc_id)
    result['gold'] = g_ent
    result['men'] = men
    return result


if __name__ == '__main__':
    q_ids = [400495.0,400496.0,400502.0,400503.0, 401235, 401267, 401917, 405178, 405921, 405991, 406096]
    q_ids = [int(q_id) for q_id in q_ids]
    for q_id in q_ids:
        const_vars_dict.res_path = os.path.join(result_path, str(q_id))
        phrase_extraction(q_id)
