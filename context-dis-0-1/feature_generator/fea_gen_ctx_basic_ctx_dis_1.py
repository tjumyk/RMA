import const_vars.constant_conll_testb
import time
import logging_config
from feature_generator.fea_gen_ctx_dis_1 import get_feature_vector as get_feature_vector_ctx_dis_1

log = logging_config.get_logger()

const_vars_dict = const_vars.constant_conll_testb.Dict_Co


def process(cases):
    """

    :param cases: a group of vectors with one positive case in the first position, [[],[],...]
    :return: feature vectors, json serialized [[],[],...]
    """

    fea_vecs = []
    s_t = time.time()
    can_size, max_prior, query_id = 0, 0, 0
    if cases:
        can_size, max_prior = get_candidate_size(cases[0][3]), get_max_men_prior(cases[0][3])
    for case in cases:
        query_id = case[0]
        fea_id = (case[0], case[2])
        fea_id = str(fea_id)
        if not const_vars_dict.redis_db_obj.is_processed_items_redis(const_vars_dict.experiment_id,
                                                                     const_vars_dict.dataset_type, fea_id):
            fea_vec = [fea_id] + get_feature_vector(case, can_size, max_prior) + get_feature_vector_ctx_dis_1(case) + \
                      [case[-1]]
            fea_vecs.append(fea_vec)
        else:
            log.info("Already processed query: {}".format(str(fea_id)))
    total_t = time.time() - s_t
    log.info("Finished Processing {}, time cost {}, with {} candidate entities. Speed: {}s per entity".format(query_id,
                                                                                                              total_t,
                                                                                                              len(
                                                                                                                  cases),
                                                                                                              total_t / len(
                                                                                                                  cases)))
    return fea_vecs


def get_feature_vector(case, can_size, max_prior):
    """

    :param case: [, , , ]
    :param can_size: the candidate size of the mention
    :param max_prior: the maximum prior of the entity
    :return: [, , , ,]
    """
    mention, entity = case[3], case[2]
    prior = float(case[1])
    feature_vec = []
    feature_vec += [prior]
    feature_vec += [can_size, max_prior, get_ent_prior(entity)]
    return feature_vec


def normalize_form(mention):
    # return mention
    return mention if len(mention) < 4 else str(mention).upper()


def get_candidate_size(men):
    size = const_vars_dict.prior_db_conn_obj.fetch_men_can_size(normalize_form(men))[0]
    return size if size else 0


def get_max_men_prior(men):
    prior = const_vars_dict.prior_db_conn_obj.fetch_max_men_prior(normalize_form(men))[0]
    return prior if prior else 0


def get_ent_prior(ent):
    return const_vars_dict.redis_db_obj.get_ents_counts_aida(ent)/const_vars_dict.ent_total_counts
