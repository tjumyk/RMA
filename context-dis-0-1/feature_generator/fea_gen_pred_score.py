import math
import re
import time

import distance as dist
import jellyfish as jf
import numpy as np

import const_vars.const_vars_abc as cop
import logging_config

log = logging_config.get_logger()
const_vars_dict = cop.Dictionaries


def process(cases):
    """

    :param cases: a group of vectors with one positive case in the first position,
            [[query_id, prior, candidate entity, mention, doc_id, begosset, 1/0 (pos/neg)],[],...]
    :return: feature vectors, json serialized [[],[],...]
    """

    fea_vecs = []
    s_t = time.time()
    can_size, max_prior, query_id = 0, 0, 0

    if cases:
        # ents = [case[2] for case in cases]
        query_id = cases[0][0]
        doc_id = cases[0][4]
        # a list of [[id, [candidate0, candidate1, ...]], ...] in a document
        # Each candidate is in the format of [entity_name, prediction_score]
        id_ents_list = const_vars_dict.redis_db_obj.get_id_ents_extended_by_doc_redis(const_vars_dict.dataset_source,
                                                                                      const_vars_dict.dataset_type,
                                                                                      doc_id)
        # id_ents_list_old_format = [[_id, candidates[0][0]] for _id, candidates in id_ents_list]
        # a set of ents in a document predicted by the basic classifier
        # ents_in_doc_set = set([x[1] for x in id_ents_list])
        id_ents_dict = dict(id_ents_list)

    for case in cases:
        fea_id = (case[0], case[2])
        fea_id = str(fea_id)
        if not const_vars_dict.redis_db_obj.is_processed_items_redis(
                const_vars_dict.experiment_id,
                const_vars_dict.dataset_source,
                const_vars_dict.dataset_type, fea_id):
            # fea_vec = [fea_id] + get_feature_vector(case, id_ents_list_old_format)
            # additional features
            # top_k_vec = get_top_k_feature_vector(case, id_ents_list)
            # fea_vec = fea_vec[0:-1] + top_k_vec + fea_vec[-1:]

            fea_vecs.append([fea_id] + get_pred_score_feature_vector(case, id_ents_dict) + [case[-1]])
        else:
            log.info("Already processed query: {}".format(str(fea_id)))
    total_t = time.time() - s_t
    log.info(
        "Finished Processing {}, time cost {}, with {} candidate entities. Speed: {}s per entity".format(
            query_id,
            total_t,
            len(
                cases),
            total_t / len(
                cases)))
    return fea_vecs


def get_pred_score_feature_vector(case, id_ents_dict: dict) -> list:
    qid = case[0]
    entity, mention, doc_id = case[2:5]

    prev_predictions = id_ents_dict.get(qid)
    # should already be sorted by score in descending order

    feature = [0.0] * 4
    scores = [score for predicted_ent, score in prev_predictions]
    if scores:
        feature[0] = scores[0]  # raw score of top prediction
        for k in range(1, 4):
            if len(scores) > k:
                feature[k] = math.exp(scores[k] - scores[0])
    return feature
