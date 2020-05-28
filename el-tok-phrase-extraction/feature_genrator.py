import phrase_extraction
import csv
import logging_config
import time
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


log = logging_config.get_logger()
config_path = database.__path__[0]
config_path = os.path.join(config_path, 'config.json')
result_path = result_folder.__path__[0]
const_vars_dict = const_vars.constant_conll_testb.Dict_Co
const_vars_dict.init_dictionaries()


def fea_gen_tf_idf_entropy_redis(q_id, w_size=50, c_size=30):
    s_t = time.time()
    const_vars_dict.db_conn_obj = database.db_kongzi3.PostgreSQL()
    const_vars_dict.prior_db_conn_obj = database.db_kongzi2.PostgreSQLAIDA()
    const_vars_dict.redis_db_obj = redis_db.redis_init.RedisConn()
    log.info("Generating entity linking queries...")
    ents, men, m_text, doc_id, g_ent = corpora.conll.get_valid_mention_entity_pairs(q_id)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    # ent_art_dict = defaultdict(str)
    ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_idf_dict, ent_tok_tf_entropy_dict, men_toks_idx = \
        spacy_tokenizer.main_gen_tf_idf_entropy_fea_redis(ents, men, m_text, doc_id)
    feas = []
    for ent_idx, ent in enumerate(ents):
        # fea = []
        etp_sum = 0
        w_etp_sum = 0
        max_etp = 0
        idf_sum = 0
        w_idf_sum = 0
        max_idf = 0
        tf_idf_sum = 0
        max_tf_idf = 0
        w_tf_idf_sum = 0
        tf_entropy_sum = 0
        max_tf_entropy = 0
        w_tf_entropy_sum = 0
        for tok in men_toks_idx.keys():
            if tok not in ent_tok_entropy_dict[ent]:
                continue
            etp_sum += ent_tok_entropy_dict[ent][tok]
            w_etp_sum += ent_tok_entropy_dict[ent][tok] * len(men_toks_idx[tok])
            max_etp = max(ent_tok_entropy_dict[ent][tok], max_etp)

            idf_sum += ent_tok_idf_dict[ent][tok]
            w_idf_sum += ent_tok_idf_dict[ent][tok] * len(men_toks_idx[tok])
            max_idf = max(ent_tok_idf_dict[ent][tok], max_idf)

            tf_idf_sum += ent_tok_tf_idf_dict[ent][tok]
            w_tf_idf_sum += ent_tok_tf_idf_dict[ent][tok] * len(men_toks_idx[tok])
            max_tf_idf = max(ent_tok_tf_idf_dict[ent][tok], max_tf_idf)

            tf_entropy_sum += ent_tok_tf_entropy_dict[ent][tok]
            w_tf_entropy_sum += ent_tok_tf_entropy_dict[ent][tok] * len(men_toks_idx[tok])
            max_tf_entropy = max(ent_tok_tf_entropy_dict[ent][tok], max_tf_entropy)
        fea = [q_id, etp_sum, w_etp_sum, max_etp, idf_sum, w_idf_sum, max_idf, tf_idf_sum, w_tf_idf_sum, max_tf_idf,
               tf_entropy_sum, w_tf_entropy_sum, max_tf_entropy, 0 if ent_idx else 1]
        feas.append(fea)
    # result['gold'] = g_ent
    # result['men'] = men
    return feas


def test_fea_combination():
    pass


if __name__ == '__main__':
    # with open('./prior_wrong.csv', 'r') as f:
    #     q_ids = [int(q_id.replace('.0', '')) for q_id in f.readline().strip().split(',')]

    with open('/Users/dzs/PycharmProjects/el-phrase-extraction/all_ids.testb', 'r') as f:
        q_ids = [int(q_id.strip()) for q_id in f.readlines()]
    # q_ids = [400495, 400496, 400500, 400500]
    feas = []
    with open('features_all.csv', 'w') as f:
        writer = csv.writer(f)
        for q_id in q_ids:
            log.info("Processing Query: {}".format(q_id))
            writer.writerows(fea_gen_tf_idf_entropy_redis(q_id))

