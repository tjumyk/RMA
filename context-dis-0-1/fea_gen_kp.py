import logging_config
import argparse
import time
from collections import defaultdict
import csv
import queue
import threading
import multiprocessing
import json
import os
import database
import database.db_kongzi3
import keyphrases.get_keyphrases
import const_vars.constant_conll_testb
import tokenizer.spacy_tokenizer
import tokenizer.pre_processing
import keyphrases.keyphrase_weight as kw
import corpora.conll
import result_folder
from multiprocessing import Pool
log = logging_config.get_logger()

config_path = database.__path__[0]
config_path = os.path.join(config_path, 'config.json')
result_path = result_folder.__path__[0]

with open(config_path, 'r') as f:
    config = json.load(f)
threads_num = config['threads_num']

log.info("Threads for processing the data: {}".format(threads_num))
const_vars_dict = const_vars.constant_conll_testb.Dict_Co

# # load constant variables from db
# log.info("Loading dictionary...")
# const_vars_dict.init_dictionaries()
# log.info("Finished Loading dictionary.")


class WrThread(threading.Thread):
    def __init__(self, wr_queue, f_wrt):
        threading.Thread.__init__(self)
        self.wr_queue = wr_queue
        self.f_wrt = f_wrt

    def wr_2_file(self, results):
        for result in results:
            self.f_wrt.writerow(result)

    def run(self):
        while True:
            results = self.wr_queue.get()
            self.wr_2_file(results)
            self.wr_queue.task_done()


class ProcessThread(threading.Thread):
    def __init__(self, in_queue, out_queue):
        threading.Thread.__init__(self)
        self.in_queue = in_queue
        self.out_queue = out_queue

    @staticmethod
    def process(cases):
        fea_vecs = []
        for case in cases:
            fea_id = (case[0], case[2])
            if not const_vars_dict.redis_db_obj.is_processed_items_redis(str(fea_id)):
                log.info("Thread {} is processing {}".format(threading.current_thread().getName(), fea_id))
                fea_vec = [fea_id] + get_feature_vector(case)
                fea_vecs.append(fea_vec)
        return fea_vecs

    def run(self):
        while True:
            cases = self.in_queue.get()
            results = self.process(cases)
            self.out_queue.put(results)
            self.in_queue.task_done()


def get_feature_vector(case):
    mention, entity = case[3], case[2]
    prior = float(case[1])
    feature_vec = []
    feature_vec += [prior]
    feature_vec += list(entity_mention_context_similarity_score_mi_idf(entity, mention, case[0]))
    return feature_vec


def entity_mention_context_similarity_score_mi_idf(entity_id, mention, doc_id, distance=1):
    s_t = time.time()
    men_keyphrase_dict = keyphrases.get_keyphrases.get_men_keyphrase_with_distance_dict(mention, doc_id, distance)
    # log.warning("Mention context generation time cost {}".format(time.time() - s_t))
    s_t = time.time()
    ent_keyphrase_dict = keyphrases.get_keyphrases.get_ent_keyphrase_with_distance_dict(entity_id, distance)
    # log.warning("Entity context generation time cost {}".format(time.time() - s_t))

    tol_mi, tol_idf, comm_token_mi, comm_token_idf = 0.0, 0.0, 0, 0
    glb_max_mi, glb_max_idf = 0.0, 0.0
    ent_keyphrases = []
    men_keyphrases = []
    for vals in ent_keyphrase_dict.values():
        ent_keyphrases += list(vals)
    for vals in men_keyphrase_dict.values():
        men_keyphrases += list(vals)

    for entity_keyphrase in ent_keyphrases:
        max_mi_sim, max_idf_sim = 0.0, 0.0
        for mention_keyphrase in men_keyphrases:
            cur_mi_similarity, cur_idf_similarity = calculate_keyphrase_similarity(entity_keyphrase, mention_keyphrase,
                                                                                   mention, entity_id)
            if cur_mi_similarity > max_mi_sim:
                comm_token_mi += 1
                max_mi_sim = cur_mi_similarity
            if cur_idf_similarity > max_idf_sim:
                comm_token_idf += 1
                max_idf_sim = cur_idf_similarity
            glb_max_mi = max(max_mi_sim, glb_max_mi)
            glb_max_idf = max(max_idf_sim, glb_max_idf)
        tol_idf += max_idf_sim
        tol_mi += max_mi_sim
    fea_vec = [tol_idf, tol_mi, comm_token_idf, comm_token_mi, glb_max_idf, glb_max_mi]
    # log.warning("Context similarity comparision time cost {}".format(time.time() - s_t))
    return fea_vec


def calculate_keyphrase_similarity(entity_keyphrase, mention_keyphrase, mention, entity_id):
    entity_keyphrase_no_mention = entity_keyphrase.replace(mention, '')
    mention_keyphrase_no_mention = mention_keyphrase.replace(mention, '')
    entity_keyphrase_tokens = tokenizer.spacy_tokenizer.tokenize_keyphrase(entity_keyphrase)
    mention_keyphrase_tokens_no_mention = tokenizer.spacy_tokenizer.tokenize_keyphrase(mention_keyphrase_no_mention)
    # entity_keyphrase_tokens_no_mention = tokenizer.spacy_tokenizer.tokenize_keyphrase(entity_keyphrase_no_mention)

    mention_tokens_index = tokenizer.pre_processing.create_index(mention_keyphrase_tokens_no_mention)
    common_words_positions = []
    common_words = []
    for token_index, token in enumerate(entity_keyphrase_tokens):
        if token in const_vars_dict.stopwords:
            continue
        word_position = find_token_in_list(mention_tokens_index, token)
        if word_position != [-1]:
            common_words_positions.extend(word_position)
            common_words.extend([token])
    intersection_size = len(set(common_words_positions))
    similarity_score_mi = 0
    similarity_score_idf = 0
    if intersection_size:
        min_cover_length = max(common_words_positions) - min(common_words_positions) + 1

        similarity_score_mi, similarity_score_idf = similarity_kernel_aida(intersection_size, min_cover_length, len(entity_keyphrase_no_mention),
                                                                           len(mention_keyphrase_tokens_no_mention), common_words,
                                                                           entity_keyphrase_tokens, entity_id)
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
        matching_words_score_idf += float(kw.get_token_weight_idf(matching_word))
        matching_words_score_mi += float(kw.get_token_weight_mi(entity_id, matching_word))
    for word in entity_keyphrase_tokens:
        # calculate the idf and mi of all words
        all_words_score_idf += float(kw.get_token_weight_idf(word))
        all_words_score_mi += float(kw.get_token_weight_mi(entity_id, word))
    score_idf = all_words_score_idf * z * ((matching_words_score_idf / all_words_score_idf) ** 2)
    score_mi = all_words_score_mi * z * ((matching_words_score_mi / all_words_score_mi) ** 2)
    # score = 0.58 * score_mi + 0.42 * score_idf
    return score_mi, score_idf


def find_token_in_list(tokens, token):
    return tokens.get(token, [-1])


def main_multithreading():
    cases_queue = queue.Queue()
    result_queue = queue.Queue()
    cases = corpora.conll.get_valid_mention_entity_pairs()
    f_handler = open(os.path.join(result_path, 'context-similarity-dis-0-1.conll'), 'w')
    writer = csv.writer(f_handler)
    for i in range(threads_num):
        t = ProcessThread(cases_queue, result_queue)
        t.setDaemon(True)
        t.start()

    t = WrThread(result_queue, writer)
    t.setDaemon(True)
    t.start()
    for case in cases:
        cases_queue.put(case)

    cases_queue.join()
    result_queue.join()
    f_handler.close()


def save_res_2_file(results):
    f_handler = open(os.path.join(result_path, 'context-similarity-dis-0-1.conll'), 'a')
    writer = csv.writer(f_handler)
    for result in results:
        writer.writerows(result)
    f_handler.close()


def process(cases):
    '''

    :param cases: a group of vectors with one positive case in the first position, [[],[],...]
    :return: feature vectors, json serialized [[],[],...]
    '''
    fea_vecs = []
    s_t = time.time()
    query_id = 0
    for case in cases:
        query_id = case[0]
        fea_id = (case[0], case[2])
        if not const_vars_dict.redis_db_obj.is_processed_items_redis(str(fea_id)):
            fea_vec = [fea_id] + get_feature_vector(case)
            fea_vecs.append(fea_vec)
        else:
            log.info("Already processed query: {}".format(str(fea_id)))
    total_t = time.time() - s_t
    log.info("Finished Processing {}, time cost {}, with {} candidate entities. Speed: {}s per entity".format(query_id,
                                                                            total_t, len(cases), total_t/len(cases)))
    return json.dumps(fea_vecs)


def main_multiprocessing(id_start, id_end, gen_all):
    s_t = time.time()
    log.info("Generating entity linking queries...")
    cases = corpora.conll.get_valid_mention_entity_pairs(id_start, id_end, gen_all)
    log.info("Finished Generation of cases, time cost {}".format(time.time()-s_t))
    results = []
    try:
        results = map(process, cases)
    except:
        log.exception("Exception occurred while processing the document.")
    finally:
        const_vars_dict.redis_db_obj.save_list_redis('result_conll_testb', *results)
        # save_res_2_file(results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generate feature vectors for entity linking cases.\n")
    parser.add_argument("x", type=int, help="The beginning index of the id for queries.", default=0)
    parser.add_argument("y", type=int, help="The ending index of the id for queries.", default=0)
    # parser.add_argument("name", type=str, help="The name of the program.", default=0)
    parser.add_argument("-a", "--all", action="store_true", help="Generate the feature vectors for all the cases.")
    args = parser.parse_args()

    # load constant variables from db
    log.info("Loading dictionary...")
    const_vars_dict.init_dictionaries()
    log.info("Finished Loading dictionary.")

    # print(gen_all)
    main_multiprocessing(args.x, args.y, args.all)
