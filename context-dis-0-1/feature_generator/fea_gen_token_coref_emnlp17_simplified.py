import re
import logging_config
import time
import const_vars.const_vars_abc as cop
import jellyfish as jf
import const_vars.const_vars_abc
import math

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
        beg_offset = cases[0][5]
        mention = cases[0][3]
        doc_ents_dict_stfd = const_vars_dict.redis_db_obj.get_ents_stfd_by_doc(doc_id, const_vars_dict.dataset_source)
        coref_men_stfd, distance = find_coreferent_ent(mention, beg_offset, doc_ents_dict_stfd)
        log.info("Get the candidate entity list and prior for stanford coreferent mention: {}".
                 format(coref_men_stfd))
        n_coref_men_stfd, candidate_with_prior_db_stfd = const_vars.const_vars_abc. \
            preprocess_mention(coref_men_stfd)
        candidate_with_prior_db_stfd = [(cop.get_redirect_entity(x[0]), x[1])
                                        for x in candidate_with_prior_db_stfd]
        ent_prior_dict_stfd = dict(candidate_with_prior_db_stfd)
        ent_prior_dict_stfd_2014 = cop.get_men_ent_priors(coref_men_stfd)

        doc_ents_dict_conll = const_vars_dict.doc_men_beg_pos_dict[doc_id]
        coref_men_conll, distance = find_coreferent_ent(mention, beg_offset, doc_ents_dict_conll)
        log.info("Get the candidate entity list and prior for conll coreferent mention: {}".
                 format(coref_men_conll))

        n_coref_men_conll, candidate_with_prior_db_conll = const_vars.const_vars_abc. \
            preprocess_mention(coref_men_conll)
        candidate_with_prior_db_conll = [(cop.get_redirect_entity(x[0]), x[1])
                                         for x in candidate_with_prior_db_conll]

        ent_prior_dict_conll = dict(candidate_with_prior_db_conll)
        ent_prior_dict_conll_2014 = cop.get_men_ent_priors(coref_men_conll)

    for case in cases:
        fea_id = (case[0], case[2])
        fea_id = str(fea_id)
        if not const_vars_dict.redis_db_obj.is_processed_items_redis(
                const_vars_dict.experiment_id,
                const_vars_dict.dataset_source,
                const_vars_dict.dataset_type, fea_id):
            fea_vec = [fea_id] + \
                      get_feature_vector(case, ent_prior_dict_stfd, ent_prior_dict_conll,
                                         ent_prior_dict_stfd_2014, ent_prior_dict_conll_2014)
            fea_vecs.append(fea_vec)
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


def get_feature_vector(case, ent_prior_dict_stfd, ent_prior_dict_conll, ent_prior_dict_stfd_2014,
                       ent_prior_dict_conll_2014):
    """
    :type ent_tok_entropy_dict: dict
    :param ent_tok_entropy_dict:
    :param ent_tok_idf_dict:
    :param case: [, , , ]
    :param can_size: the candidate size of the mention
    :param max_prior: the maximum prior of the entity
    :return: [, , , ,]
    """
    entity, mention, doc_id = case[2:5]
    prior = float(case[1])
    max_prior_entity = const_vars_dict.max_ent_priors_dict_emnlp17[doc_id].get(entity, prior)
    max_prior_entity_2014 = const_vars_dict.max_ent_priors_dict_2014[doc_id].get(entity, prior)
    # get the coreferent mention recognized by the stanford NER
    prior_conll_coref = ent_prior_dict_conll.get(entity, 0) if ent_prior_dict_conll else prior
    prior_stfd_ner_coref = ent_prior_dict_stfd.get(entity, 0) if ent_prior_dict_stfd else prior

    prior_conll_coref_2014 = ent_prior_dict_conll_2014.get(entity, 0) \
        if ent_prior_dict_conll_2014 else prior
    prior_stfd_ner_coref_2014 = ent_prior_dict_stfd_2014.get(entity, 0) \
        if ent_prior_dict_stfd_2014 else prior

    feature_vec = []
    feature_vec += [max_prior_entity_2014, prior_stfd_ner_coref_2014, prior_conll_coref_2014] + \
                    str_sim_fea_vec(mention, entity) + [case[-1]]
    return feature_vec


def normalize_form(mention):
    # return mention
    return mention if len(mention) < 4 else str(mention).upper()


def str_sim_fea_vec(mention, entity, options=None):
    # transfer to lower letter, '_' to space
    if not options:
        options = ['ES', 'JARO-WINKLER', 'START-WITH', 'END-WITH', 'SAME']
    mention_ = re.sub(r"[^a-zA-Z0-9_]+", "_",
                      mention.lower().split('(')[0].strip())
    entity_ = re.sub(r"[^a-zA-Z0-9_]+", "_",
                     entity.lower().split('(')[0].strip())
    if len(mention_) > len(entity_):
        sim_fea_vec = [0.0 for _ in options]
    else:
        sim_fea_vec = []
        for opt in options:
            sim_fea_vec.append(cal_str_similarity(mention_, entity_, opt))

    return sim_fea_vec


def cal_str_similarity(str_1, str_2, option):
    multiset_1 = str_1.split()
    multiset_2 = str_2.split()
    # Jaccard similarity
    # if option == 'JACC':
    #     return 1.0 - dist.jaccard(multiset_1, multiset_2)
    # Cosine similarity
    if option == 'COS':
        comm_len = len([word for word in multiset_1 if word in multiset_2])
        return comm_len * 1.0 / math.sqrt(len(multiset_1) * len(multiset_2))
    # Dice similarity
    elif option == 'DICE':
        comm_len = len([word for word in multiset_1 if word in multiset_2])
        return comm_len * 2.0 / (len(multiset_1) + len(multiset_2))
    # Edit similarity
    elif option == 'ES':
        return 1.0 - jf.levenshtein_distance(str_1, str_2) * 1.0 / max(
            len(str_1), len(str_2))
    # Hamming similarity
    elif option == 'HAMMING':
        return 1.0 - jf.hamming_distance(str_1, str_2) * 1.0 / max(len(str_1),
                                                                   len(str_2))
    # Jaro distance
    elif option == 'JARO':
        return jf.jaro_distance(str_1, str_2)
    # Jaro-Winkler distance
    elif option == 'JARO-WINKLER':
        return jf.jaro_winkler(str_1, str_2)
    # Overlap similarity
    elif option == 'OVERLAP':
        comm_len = len([word for word in multiset_1 if word in multiset_2])
        return comm_len * 1.0 / max(len(multiset_1), len(multiset_2))
    # entity string start with mention string
    elif option == 'START-WITH':
        return str_2.startswith(str_1)

    elif option == 'END-WITH':
        return str_2.endswith(str_1)

    elif option == 'SAME':
        return str_1 == str_2


def start_or_end_with(entity, mention, ignore_case=True):
    return (entity.lower().startswith(mention.lower()) or entity.lower().endswith(mention.lower())) \
        if ignore_case else (entity.startswith(mention) or entity.endswith(mention))


def find_coreferent_ent(men: str, beg_pos: int, doc_ents_dict: dict) -> tuple:
    nearest_ent = men
    prev_dist = None
    curt_dist = None
    for key, val in doc_ents_dict.items():
        if len(men) < len(key) and start_or_end_with(key, men):
            for idx in val:
                curt_dist = beg_pos - int(idx)
                if not curt_dist:
                    return key, curt_dist
                elif prev_dist is None:
                    nearest_ent = key
                    prev_dist = curt_dist
                else:
                    if prev_dist * curt_dist > 0:
                        if abs(prev_dist) > abs(curt_dist):
                            prev_dist = curt_dist
                            nearest_ent = key
                    elif prev_dist < 0 < curt_dist:
                        prev_dist = curt_dist
                        nearest_ent = key
    return nearest_ent, curt_dist
