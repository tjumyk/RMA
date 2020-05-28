import logging_config
import time
import const_vars.const_vars_abc as cop
from tokenizer import spacy_tokenizer

log = logging_config.get_logger()
const_vars_dict = cop.Dictionaries


def process(cases):
    """

    :param cases: a group of vectors with one positive case in the first position,
            [[query_id, prior, candidate entity, mention, doc_id, 1/0 (pos/neg)],[],...]
    :return: feature vectors, json serialized [[],[],...]
    """

    fea_vecs = []
    s_t = time.time()
    can_size, max_prior, query_id = 0, 0, 0
    if cases:
        ent_priors_dict = cop.get_men_ent_priors(cases[0][3])
        ents = [case[2] for case in cases]
        # ents_count_emnlp17 = [const_vars_dict.redis_db_obj.get_ents_counts_emnlp17(ent) for ent in ents]
        ents_count_2014 = [cop.get_entity_counts(ent) for ent in ents]
        # sum_counts_emnlp17 = sum(ents_count_emnlp17)
        # sum_counts_emnlp17 = sum_counts_emnlp17 if sum_counts_emnlp17 else 1
        sum_counts_2014 = sum(ents_count_2014)
        sum_counts_2014 = sum_counts_2014 if sum_counts_2014 else 1
        query_id = cases[0][0]
        doc_id = cases[0][4]
        can_size, max_prior_emnlp17 = get_candidate_size_and_max_men_prior(cases[0][3])
        max_prior_2014 = max(ent_priors_dict.values()) if ent_priors_dict else 0
        ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_dict, men_toks_idx = \
            spacy_tokenizer.main_gen_tf_idf_entropy_fea_redis(ents, query_id, doc_id, cases[0][3])
        # const_vars_dict.redis_db_obj.save_dict_redis('entropy', cases[0][3],
        #                                              json.dumps(ent_tok_entropy_dict))
        # const_vars_dict.redis_db_obj.save_dict_redis('idf', cases[0][3],
        #                                              json.dumps(ent_tok_idf_dict))
        # const_vars_dict.redis_db_obj.save_dict_redis('tf',
        #                                              json.loads(ent_tok_entropy_dict))
    for case_idx, case in enumerate(cases):
        fea_id = (case[0], case[2])
        fea_id = str(fea_id)
        if not const_vars_dict.redis_db_obj.\
                is_processed_items_redis(
                                const_vars_dict.experiment_id, const_vars_dict.dataset_source,
                                const_vars_dict.dataset_type, fea_id):
            fea_vec = [fea_id] + \
                      [ent_priors_dict.get(case[2], 0), max_prior_2014, len(ent_priors_dict)] + \
                      get_feature_vector(case, can_size, max_prior,
                                                    ent_tok_entropy_dict, ent_tok_idf_dict,
                                                    ent_tok_tf_dict, men_toks_idx,
                                                    ents_count_2014[case_idx],
                                                    sum_counts_2014)
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


def fea_gen_tf_idf_entropy_redis(ent, ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_dict,
                                 men_toks_idx, w_size=50, c_size=30):
    etp_sum = 0
    w_etp_sum = 0
    min_etp = 0
    idf_sum = 0
    w_idf_sum = 0

    tf_idf_sum = 0
    max_tf_idf = 0
    w_tf_idf_sum = 0
    # tf_entropy_sum = 0
    # max_tf_entropy = 0
    # w_tf_entropy_sum = 0
    tf_idf_entropy_sum = 0
    tf_entropy_sum = 0

    if ent not in ent_tok_idf_dict:
        return [etp_sum, w_etp_sum, min_etp, idf_sum, w_idf_sum, 0, tf_idf_sum, w_tf_idf_sum, max_tf_idf,
                tf_idf_entropy_sum, tf_entropy_sum]
    idfs = [val for key, val in ent_tok_idf_dict[ent].items() if key in men_toks_idx.keys()]
    (max_idf, min_idf) = (max(idfs), min(idfs)) if idfs else (0, 0)

    for tok in men_toks_idx.keys():
        if tok not in ent_tok_entropy_dict[ent]:
            continue
        etp_sum += ent_tok_entropy_dict[ent][tok]
        w_etp_sum += ent_tok_entropy_dict[ent][tok] * len(men_toks_idx[tok])
        min_etp = min(ent_tok_entropy_dict[ent][tok], min_etp)

        idf_sum += ent_tok_idf_dict[ent][tok]
        w_idf_sum += ent_tok_idf_dict[ent][tok] * len(men_toks_idx[tok])

        tf_idf = ent_tok_idf_dict[ent][tok] * ent_tok_tf_dict[ent][tok]
        tf_idf_sum += tf_idf
        w_tf_idf_sum += tf_idf * len(men_toks_idx[tok])
        max_tf_idf = max(tf_idf, max_tf_idf)

        tf_idf_entropy = tf_idf * ent_tok_entropy_dict[ent][tok]
        tf_idf_entropy_sum += tf_idf_entropy

        tf_entropy_sum += ent_tok_entropy_dict[ent][tok] * ent_tok_tf_dict[ent][tok]

    feas = [etp_sum, w_etp_sum, min_etp, idf_sum, w_idf_sum, max_idf, tf_idf_sum, w_tf_idf_sum, max_tf_idf,
            tf_idf_entropy_sum, tf_entropy_sum]
    return feas


def get_feature_vector(case, can_size, max_prior, ent_tok_entropy_dict, ent_tok_idf_dict,
                       ent_tok_tf_dict, men_toks_idx, ent_count_2014,
                       ent_count_sum_2014):
    """

    :type ent_tok_entropy_dict: dict
    :param ent_tok_entropy_dict:
    :param ent_tok_idf_dict:
    :param case: [, , , ]
    :param can_size: the candidate size of the mention
    :param max_prior: the maximum prior of the entity
    :return: [, , , ,]
    """
    mention, entity = case[3], case[2]
    prior = float(case[1])
    feature_vec = []
    feature_vec += [prior]
    feature_vec += [can_size, max_prior] + \
                   [ent_count_2014 / ent_count_sum_2014,
                    ent_count_2014 / const_vars_dict.ent_total_counts_2014] + \
                    fea_gen_tf_idf_entropy_redis(entity, ent_tok_entropy_dict, ent_tok_idf_dict,
                                                                        ent_tok_tf_dict,
                                                                        men_toks_idx) + \
                   [case[-1]]
    return feature_vec


def normalize_form(mention):
    # return mention
    return mention if len(mention) < 4 else str(mention).upper()


def get_candidate_size_and_max_men_prior(men):
    cur_m, can_ents = cop.preprocess_mention(men)
    prior = can_ents[0][1] if can_ents else 0
    return len(can_ents), prior


def get_max_men_prior(men):
    prior = const_vars_dict.prior_db_conn_obj.fetch_max_men_prior(normalize_form(men))[0]
    return prior if prior else 0


def get_ent_prior(ent):
    return const_vars_dict.redis_db_obj.get_ents_counts_emnlp17(ent) / \
           const_vars_dict.ent_total_counts
