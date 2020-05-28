import re
import logging_config
import time
import const_vars.const_vars_abc as cop
import jellyfish as jf
import distance as dist
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
        # a list of [[id, ent], ...] in a document
        id_ents_list = const_vars_dict.redis_db_obj.get_id_ents_by_doc_redis(const_vars_dict.dataset_source,
                                                                             const_vars_dict.dataset_type,
                                                                             doc_id)
        # a set of ents in a document predicted by the basic classifier
        # ents_in_doc_set = set([x[1] for x in id_ents_list])

    for case in cases:
        fea_id = (case[0], case[2])
        fea_id = str(fea_id)
        if not const_vars_dict.redis_db_obj.is_processed_items_redis(
                const_vars_dict.experiment_id,
                const_vars_dict.dataset_source,
                const_vars_dict.dataset_type, fea_id):
            fea_vec = [fea_id] + get_feature_vector(case, id_ents_list)
            fea_vecs.append(fea_vec)
        else:
            log.info("Already processed query: {}".format(str(fea_id)))
    total_t = time.time() - s_t
    #log.info(
    #    "Finished Processing {}, time cost {}, with {} candidate entities. Speed: {}s per entity".format(
    #        query_id,
    #        total_t,
    #        len(
    #            cases),
    #        total_t / len(
    #            cases)))
    return fea_vecs


def get_feature_vector(case, id_ents_list, use_ext=True):
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
    # filter the coherent ents, only different ents are left
    coh_ents = set([x[1] for x in id_ents_list if x[1] != entity and x[0] != case[0]])
    # feature vector, including 9 features, corresponding to the Local and Global Algorithms ...
    feature_vec_tal = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    feature_vec_max = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    if use_ext:
        feature_vec_tal += [0, 0, 0, 0]
        feature_vec_max += [0, 0, 0, 0]
    # feature vector, str similarity
    feature_vec_tal_str = [0, 0]
    feature_vec_max_str = [0, 0]
    # coh_ent_count_unidirection = 0
    # coh_ent_count_bidirection = 0

    ents_s_inlinks = get_links_by_ent(entity)
    ents_s_outlinks = get_links_by_ent(entity, link_type='outlinks')

    for coh_ent in coh_ents:
        cur_feas_str = str_sim_fea_vec(entity, coh_ent)
        feature_vec_tal_str[0] += cur_feas_str[0]
        feature_vec_tal_str[1] += cur_feas_str[1]
        feature_vec_max_str[0] = max(cur_feas_str[0], feature_vec_max_str[0])
        feature_vec_max_str[1] = max(cur_feas_str[1], feature_vec_max_str[1])

        pair_info = const_vars_dict.redis_db_obj.get_ent_pair_info(entity, coh_ent)
        ents_t_inlinks = None
        ents_t_outlinks = None

        if pair_info is None:
            ents_t_inlinks = get_links_by_ent(coh_ent)
            ents_t_outlinks = get_links_by_ent(coh_ent, link_type='outlinks')

            cur_pmi_in = pmi_similarity(ents_s_inlinks, ents_t_inlinks)
            cur_ngd_in = ngd_similarity(ents_s_inlinks, ents_t_inlinks)
            cur_pmi_out = pmi_similarity(ents_s_outlinks, ents_t_outlinks)
            cur_ngd_out = ngd_similarity(ents_s_outlinks, ents_t_outlinks)

            has_link = check_links_between_ents(entity, coh_ent, ents_s_inlinks, ents_t_inlinks)
            if has_link:
                # coh_ent_count_unidirection += 1
                cur_feas = [cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out]

                for feature_vec_max_idx in range(4):
                    feature_vec_max[feature_vec_max_idx] = \
                        max(feature_vec_max[feature_vec_max_idx], cur_feas[feature_vec_max_idx])
                    feature_vec_tal[feature_vec_max_idx] += cur_feas[feature_vec_max_idx]

            has_bidirectional_link = check_links_between_ents(entity, coh_ent, ents_s_inlinks, ents_t_inlinks,
                                                              bidirection=True)
            if has_bidirectional_link:
                # coh_ent_count_bidirection += 1
                cur_feas = [1, cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out]

                for feature_vec_max_idx in range(4, 9):
                    feature_vec_max[feature_vec_max_idx] = \
                        max(feature_vec_max[feature_vec_max_idx], cur_feas[feature_vec_max_idx - 4])
                    feature_vec_tal[feature_vec_max_idx] += cur_feas[feature_vec_max_idx - 4]

            pair_info = cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out, has_link, has_bidirectional_link
            const_vars_dict.redis_db_obj.save_ent_pair_info(entity, coh_ent, pair_info)
        else:
            cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out, has_link, has_bidirectional_link = pair_info
            if has_link:
                cur_feas = [cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out]
                for feature_vec_max_idx in range(4):
                    feature_vec_max[feature_vec_max_idx] = \
                        max(feature_vec_max[feature_vec_max_idx], cur_feas[feature_vec_max_idx])
                    feature_vec_tal[feature_vec_max_idx] += cur_feas[feature_vec_max_idx]

            if has_bidirectional_link:
                cur_feas = [1, cur_pmi_in, cur_ngd_in, cur_pmi_out, cur_ngd_out]
                for feature_vec_max_idx in range(4, 9):
                    feature_vec_max[feature_vec_max_idx] = \
                        max(feature_vec_max[feature_vec_max_idx], cur_feas[feature_vec_max_idx - 4])
                    feature_vec_tal[feature_vec_max_idx] += cur_feas[feature_vec_max_idx - 4]

        if use_ext:
            pair_info_ext = const_vars_dict.redis_db_obj.get_ent_pair_info_ext(entity, coh_ent)
            if pair_info_ext is None:
                if ents_t_inlinks is None:
                    ents_t_inlinks = get_links_by_ent(coh_ent)
                if ents_t_outlinks is None:
                    ents_t_outlinks = get_links_by_ent(coh_ent, link_type='outlinks')

                cur_njs_in = njs_similarity(ents_s_inlinks, ents_t_inlinks)
                cur_njs_out = njs_similarity(ents_s_outlinks, ents_t_outlinks)

                pair_info_ext = cur_njs_in, cur_njs_out
                const_vars_dict.redis_db_obj.save_ent_pair_info_ext(entity, coh_ent, pair_info_ext)
            else:
                cur_njs_in, cur_njs_out = pair_info_ext

            if has_link:
                feature_vec_max[9] = max(feature_vec_max[9], cur_njs_in)
                feature_vec_max[10] = max(feature_vec_max[10], cur_njs_out)
                feature_vec_tal[9] += cur_njs_in
                feature_vec_tal[10] += cur_njs_out

            if has_bidirectional_link:
                feature_vec_max[11] = max(feature_vec_max[11], cur_njs_in)
                feature_vec_max[12] = max(feature_vec_max[12], cur_njs_out)
                feature_vec_tal[11] += cur_njs_in
                feature_vec_tal[12] += cur_njs_out


    feature_vec = []
    if coh_ents:
        feature_vec += feature_vec_max + [x / len(coh_ents) for x in feature_vec_tal] + \
                       feature_vec_max_str + [x / len(coh_ents) for x in feature_vec_tal_str] + \
                       [case[-1]]
    else:
        feature_vec += feature_vec_max + feature_vec_tal + feature_vec_max_str + \
                       feature_vec_tal_str + [case[-1]]

    return feature_vec

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


# Normalized Jaccard Similarity
def njs_similarity(ents_s, ents_t):
    ss = set(ents_s)
    st = set(ents_t)
    return math.log(len(ss & st) + 1) / math.log(len(set.union(ss, st)) + 1)

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
