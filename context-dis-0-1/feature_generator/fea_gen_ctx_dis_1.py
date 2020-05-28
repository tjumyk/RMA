import const_vars.constant_conll_testb
import tokenizer.spacy_tokenizer
import tokenizer.pre_processing
import keyphrases_redis.get_keyphrases_redis
import keyphrases_redis.keyphrase_weight_redis as kw_redis
import time
import logging_config


log = logging_config.get_logger()

const_vars_dict = const_vars.constant_conll_testb.Dict_Co


def get_feature_vector(case):
    mention, entity = case[3], case[2]
    prior = float(case[1])
    feature_vec = []
    feature_vec += [prior]
    feature_vec += list(entity_mention_context_similarity_score_mi_idf(entity, mention, case[0]))
    return feature_vec


def entity_mention_context_similarity_score_mi_idf(entity_id, mention, doc_id, distance=1):
    # s_t = time.time()
    men_keyphrase_dict = keyphrases_redis.get_keyphrases_redis.get_men_keyphrase_with_distance_dict(mention, doc_id)
    # log.warning("Mention context generation time cost {}".format(time.time() - s_t))
    # s_t = time.time()
    ent_keyphrase_dict = keyphrases_redis.get_keyphrases_redis.get_ent_keyphrase_with_distance_dict(entity_id, distance)
    # log.warning("Entity context generation time cost {}".format(time.time() - s_t))
    mention_lemma = tokenizer.spacy_tokenizer.tokenize_keyphrase(mention)[0]
    tol_mi, tol_idf, comm_token_mi, comm_token_idf = 0.0, 0.0, 0, 0
    glb_max_mi, glb_max_idf = 0.0, 0.0
    ent_keyphrases = []
    men_keyphrases = []
    for vals in ent_keyphrase_dict.values():
        ent_keyphrases += list(vals)
    for vals in men_keyphrase_dict.values():
        men_keyphrases += list(vals)

    for entity_keyphrase in ent_keyphrases:
        entity_keyphrase_tokens = tokenizer.spacy_tokenizer.tokenize_keyphrase(entity_keyphrase)
        entity_keyphrase_tokens_no_mention = [e_toks for e_toks in entity_keyphrase_tokens if e_toks != mention_lemma and
                                              e_toks not in const_vars_dict.stopwords]
        if not entity_keyphrase_tokens_no_mention:
            continue
        max_mi_sim, max_idf_sim = 0.0, 0.0
        for mention_keyphrase in men_keyphrases:
            mention_keyphrase_tokens = tokenizer.spacy_tokenizer.tokenize_keyphrase(mention_keyphrase)
            # mention_lemma = tokenizer.spacy_tokenizer.tokenize_keyphrase(mention)

            mention_keyphrase_tokens_no_mention = [m_toks for m_toks in mention_keyphrase_tokens if
                                                   m_toks != mention_lemma and
                                                   m_toks not in const_vars_dict.stopwords]
            if not mention_keyphrase_tokens_no_mention:
                continue
            cur_mi_similarity, cur_idf_similarity = calculate_keyphrase_similarity(entity_keyphrase_tokens_no_mention,
                                                                                   mention_keyphrase_tokens_no_mention,
                                                                                   mention_lemma, entity_id)
            if cur_mi_similarity > max_mi_sim:
                comm_token_mi += 1
                max_mi_sim = cur_mi_similarity
                glb_max_mi = max(max_mi_sim, glb_max_mi)
            if cur_idf_similarity > max_idf_sim:
                comm_token_idf += 1
                max_idf_sim = cur_idf_similarity
                glb_max_idf = max(max_idf_sim, glb_max_idf)
        tol_idf += max_idf_sim
        tol_mi += max_mi_sim
    fea_vec = [tol_idf, tol_mi, comm_token_idf, comm_token_mi, glb_max_idf, glb_max_mi]
    # log.warning("Context similarity comparision time cost {}".format(time.time() - s_t))
    return fea_vec


def calculate_keyphrase_similarity(entity_keyphrase_tokens_no_mention, mention_keyphrase_tokens_no_mention,
                                   mention, entity_id):
    mention_tokens_index = tokenizer.pre_processing.create_index(mention_keyphrase_tokens_no_mention)
    common_words_positions = []
    common_words = []
    for token_index, token in enumerate(entity_keyphrase_tokens_no_mention):
        # if token in const_vars_dict.stopwords:
        #     continue
        word_position = find_token_in_list(mention_tokens_index, token)
        if word_position != [-1]:
            common_words_positions.extend(word_position)
            common_words.append(token)
    intersection_size = len(set(common_words_positions))
    similarity_score_mi, similarity_score_idf = 0, 0
    if intersection_size:
        min_cover_length = max(common_words_positions) - min(common_words_positions) + 1

        similarity_score_mi, similarity_score_idf = similarity_kernel_aida(intersection_size, min_cover_length,
                                                                           len(entity_keyphrase_tokens_no_mention),
                                                                           len(mention_keyphrase_tokens_no_mention),
                                                                           common_words,
                                                                           entity_keyphrase_tokens_no_mention, entity_id)
    # print(similarity_score)
    # log.info("Similarity score: %s " % similarity_score)
    return similarity_score_mi, similarity_score_idf


def similarity_kernel_aida(matching_words_size, min_cover_length, entity_length_no_mention, mention_length,
                           matching_words, entity_keyphrase_tokens, entity_id):
    z = matching_words_size / min_cover_length
    matching_words_score_idf = 0
    all_words_score_idf = 0
    matching_words_score_mi = 0
    all_words_score_mi = 0
    for matching_word in matching_words:
        # calculate the idf and mi of matching word
        matching_words_score_idf += float(kw_redis.get_token_weight_idf(matching_word))
        matching_words_score_mi += float(kw_redis.get_token_weight_mi(entity_id, matching_word))
    for word in entity_keyphrase_tokens:
        # calculate the idf and mi of all words
        all_words_score_idf += float(kw_redis.get_token_weight_idf(word))
        all_words_score_mi += float(kw_redis.get_token_weight_mi(entity_id, word))
    score_idf = all_words_score_idf * z * ((matching_words_score_idf / all_words_score_idf) ** 2)
    score_mi = all_words_score_mi * z * ((matching_words_score_mi / all_words_score_mi) ** 2)
    # score = 0.58 * score_mi + 0.42 * score_idf
    return score_mi, score_idf


def find_token_in_list(tokens, token):
    return tokens.get(token, [-1])


def process(cases):
    """

    :param cases: a group of vectors with one positive case in the first position, [[],[],...]
    :return: feature vectors, json serialized [[],[],...]
    """

    fea_vecs = []
    s_t = time.time()
    query_id = 0
    for case in cases:
        query_id = case[0]
        fea_id = (case[0], case[2])
        fea_id = str(fea_id)
        if not const_vars_dict.redis_db_obj.is_processed_items_redis(const_vars_dict.experiment_id,
                                                                     const_vars_dict.dataset_type, fea_id):
            fea_vec = [fea_id] + get_feature_vector(case)
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
