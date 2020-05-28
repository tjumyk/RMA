import re
import logging_config
import time
import const_vars.const_vars_abc as cop
import jellyfish as jf
import distance as dist
import math
import numpy as np

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

    for case in cases:
        fea_id = (case[0], case[2])
        fea_id = str(fea_id)
        if not const_vars_dict.redis_db_obj.is_processed_items_redis(
                const_vars_dict.experiment_id,
                const_vars_dict.dataset_source,
                const_vars_dict.dataset_type, fea_id):
            # fea_vec = [fea_id] + get_feature_vector(case, id_ents_list_old_format)
            top_k_vec = get_top_k_feature_vector(case, id_ents_list)
            fea_vec = [fea_id] + top_k_vec + [case[-1]]
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


def get_top_k_feature_vector(case, id_ents_list, top_k=3) ->list:
    qid = case[0]
    entity, mention, doc_id = case[2:5]

    ents_s_inlinks = None  # will be lazy-loaded
    ents_s_outlinks = None  # will be lazy-loaded

    coh_entities = {candidate[0] for _qid, candidates in id_ents_list for candidate in candidates} - {entity}
    coh_features = {}
    for coh_ent in coh_entities:
        pair_info = const_vars_dict.redis_db_obj.get_ent_pair_info(entity, coh_ent)
        if pair_info is None:
            ents_t_inlinks = get_links_by_ent(coh_ent)
            ents_t_outlinks = get_links_by_ent(coh_ent, link_type='outlinks')

            if ents_s_inlinks is None:
                ents_s_inlinks = get_links_by_ent(entity)
            if ents_s_outlinks is None:
                ents_s_outlinks = get_links_by_ent(entity, link_type='outlinks')
            cur_pmi_in = pmi_similarity(ents_s_inlinks, ents_t_inlinks)
            cur_ngd_in = ngd_similarity(ents_s_inlinks, ents_t_inlinks)
            cur_pmi_out = pmi_similarity(ents_s_outlinks, ents_t_outlinks)
            cur_ngd_out = ngd_similarity(ents_s_outlinks, ents_t_outlinks)

            has_link = check_links_between_ents(entity, coh_ent, ents_s_inlinks, ents_t_inlinks)
            has_bidirectional_link = check_links_between_ents(entity, coh_ent, ents_s_inlinks, ents_t_inlinks,
                                                              bidirection=True)
            pair_info = cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out, has_link, has_bidirectional_link
            const_vars_dict.redis_db_obj.save_ent_pair_info(entity, coh_ent, pair_info)

        cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out, has_link, has_bidirectional_link = pair_info
        cur_feas_link = [0] * 9
        if has_link:
            cur_feas_link[0:4] = [cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out]
        if has_bidirectional_link:
            cur_feas_link[4:] = [1, cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out]

        cur_feas_str = str_sim_fea_vec(entity, coh_ent)

        cur_feas = cur_feas_link + cur_feas_str
        coh_features[coh_ent] = cur_feas

    max_candidate_coh_features = []
    avg_candidate_coh_features = []
    for coh_qid, candidates in id_ents_list:
        if qid == coh_qid:
            continue
        candidate_coh_features = [coh_features[coh_ent] for coh_ent, score in candidates[:top_k] if coh_ent != entity]
        if candidate_coh_features:
            candidate_coh_features = np.array(candidate_coh_features)
            max_candidate_coh_features.append(np.max(candidate_coh_features, axis=0))
            avg_candidate_coh_features.append(np.average(candidate_coh_features, axis=0))

    
    if max_candidate_coh_features:
        max_candidate_coh_features = np.vstack(max_candidate_coh_features)
        max_max_features = np.max(max_candidate_coh_features, axis=0)
        avg_max_features = np.average(max_candidate_coh_features, axis=0)
    else:
        max_max_features = np.zeros(9 + 2)
        avg_max_features = np.zeros(9 + 2)

    if avg_candidate_coh_features:
        avg_candidate_coh_features = np.vstack(avg_candidate_coh_features)
        max_avg_features = np.max(avg_candidate_coh_features, axis=0)
        avg_avg_features = np.average(avg_candidate_coh_features, axis=0)
    else:
        max_avg_features = np.zeros(9 + 2)
        avg_avg_features = np.zeros(9 + 2)
    
    return np.concatenate([max_max_features, avg_max_features, max_avg_features, avg_avg_features]).tolist()



def normalize_form(mention):
    # return mention
    return mention if len(mention) < 4 else str(mention).upper()


def str_sim_fea_vec(ent_s, ent_t, options=None):
    # transfer to lower letter, '_' to space
    if not options:
        options = ['ES', 'JACC']

    ent_s_str = re.sub(r"[_().\", /\']+", " ", ent_s.lower()).strip()
    ent_t_str = re.sub(r"[_().\", /\']+", " ", ent_t.lower()).strip()
    sim_fea_vec = []
    for opt in options:
        sim_fea_vec.append(cal_str_similarity(ent_s_str, ent_t_str, opt))

    return sim_fea_vec


def cal_str_similarity(str_1, str_2, option):
    multiset_1 = str_1.split()
    multiset_2 = str_2.split()
    # Jaccard similarity
    if option == 'JACC':
        return 1.0 - dist.jaccard(multiset_1, multiset_2)
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


def check_links_between_ents(ent_1, ent_2, inlinks_ent_1, inlinks_ent_2, bidirection=False):
    """
    Check whether there is a link between two ents
    :param ent_1:
    :param ent_2:
    :param bidirection:
    :return:
    """
    return (ent_1 in inlinks_ent_2 or ent_2 in inlinks_ent_1) if not bidirection else \
        (ent_1 in inlinks_ent_2 and ent_2 in inlinks_ent_1)


def get_links_by_ent(ent, link_type='inlinks'):
    wiki_pre_str = 'en.wikipedia.org/wiki/'
    if link_type == 'inlinks':
        #log.info('Fetching inlinks from redis...')
        inlinks_ent = const_vars_dict.redis_db_obj.fetch_inlinks_redis(ent, link_type='inlinks')
        #log.info('Inlinks fetched from redis')
        if not inlinks_ent and not const_vars_dict.redis_db_obj.has_inlinks_redis(ent):
            log.info("PostgreSQL: fetching inlinks for entity {}...".format(ent))
            wiki_ents = wiki_pre_str + ent
            inlinks_ent_db = const_vars_dict.prior_db_conn_obj.fetch_inlinks_by_ent(wiki_ents)
            inlinks_ent = [x[0].replace(wiki_pre_str, '') for x in inlinks_ent_db]
            log.info("Redis: caching inlinks for entity {}...".format(ent))
            const_vars_dict.redis_db_obj.save_inlinks_redis(ent, inlinks_ent)
            #log.info("Inlinks cached in redis")
        return inlinks_ent
    if link_type == 'outlinks':
        #log.info('Fetching outlinks from redis...')
        outlinks_ent = const_vars_dict.redis_db_obj.fetch_outlinks_redis(ent)
        #log.info('Outlinks fetched from redis')
        if not outlinks_ent and not const_vars_dict.redis_db_obj.has_outlinks_redis(ent):
            log.info("PostgreSQL: fetching outlinks for entity {}...".format(ent))
            wiki_ents = wiki_pre_str + ent
            outlinks_ent_db = const_vars_dict.prior_db_conn_obj.fetch_outlinks_by_ent(wiki_ents)
            outlinks_ent = [x[0].replace(wiki_pre_str, '') for x in outlinks_ent_db]
            log.info("Redis: caching outlinks for entity {}...".format(ent))
            const_vars_dict.redis_db_obj.save_outlinks_redis(ent, outlinks_ent)
            #log.info("Outlinks cached in redis")
        return outlinks_ent


# Normalized Google Distance
def ngd_similarity(ents_s, ents_t, index_size=6274625):
    """
    Calculate the normalized google distance similarity, 1 - ngd
    :param ents_s:
    :param ents_t:
    :param index_size:
    :return:
    """
    ent_sets_s = set(ents_s)
    ent_sets_t = set(ents_t)
    min_links, max_links = min(len(ent_sets_s), len(ent_sets_t)), \
                           max(len(ent_sets_s), len(ent_sets_t))
    com_links = len(ent_sets_s & ent_sets_t)
    if min_links and max_links and com_links:
        return 1 - (math.log(max_links) - math.log(com_links)) / \
               (math.log(index_size) - math.log(min_links))
    else:
        return 0


# PMI
def pmi_similarity(ents_s, ents_t, index_size=6274625, normalize=True):
    ent_sets_s = set(ents_s)
    ent_sets_t = set(ents_t)
    s_links, t_links = len(ent_sets_s), len(ent_sets_t)
    com_links = len(ent_sets_s & ent_sets_t)
    p_s = s_links / index_size
    p_t = t_links / index_size
    p_c = com_links / index_size
    # print(p_s, p_t, p_c)
    if p_s and p_t and p_c:
        return p_c / (p_s * p_t) if not normalize else p_c / (p_s * p_t) / min(1 / p_s, 1 / p_t)
    else:
        return 0
