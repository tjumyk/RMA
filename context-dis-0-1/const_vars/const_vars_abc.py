"""An abstract class for constant variables"""

import redis_db.redis_init
import result_folder
import abc
import json
import database
import hashlib
import os
import database.db_kongzi2
import database.db_kongzi3
from collections import defaultdict
import traceback
import re
import logging_config
import spacy
import csv
import requests

result_path = result_folder.__path__[0]

log = logging_config.get_logger()


class Dictionaries(object):
    __metaclass__ = abc.ABCMeta

    collection_size = 0
    p_name = 'undefined'
    m_start = 0
    m_end = 0
    ent_repo_id = 'ent-tok-ner-spacy'
    dataset_source = None
    dataset_type = None
    word_idf_dict = {}
    word_mi_dict = {}
    stopwords = []
    processed_entities = []
    men_ent_prior_dict = defaultdict(float)
    men_ent_dict = defaultdict(set)
    spacy_init = None
    processed_items = []
    men_keyphrase_dict = defaultdict(list)
    ent_keyphrase_dict = defaultdict(list)
    db_conn_obj = database.db_kongzi3.PostgreSQL()
    prior_db_conn_obj = database.db_kongzi2.PostgreSQLAIDA()
    redis_db_obj = redis_db.redis_init.RedisConn()
    candidate_size = 50
    experiment_id = ''
    ent_total_counts = 31580076297
    ent_total_counts_2014 = 139239040
    max_ent_priors_dict_emnlp17, max_ent_priors_dict_2014 = None, None
    id_doc_dict, doc_men_dict, doc_men_beg_pos_dict = None, None, None

    @staticmethod
    def init_dictionaries():
        log.info("Loading words IDF...")
        # Dictionaries.word_idf_dict = Dictionaries.db_conn_obj.get_all_words_idf_dict()
        # log.info("Loading entity words MI...")
        # Dictionaries.word_mi_dict = Dictionaries.db_conn_obj.get_all_words_mi_dict()
        # log.info("Loading collection size...")
        # Dictionaries.collection_size = Dictionaries.redis_db_obj.get_collection_size()
        # log.info("Loading Stopwords...")
        # Dictionaries.stopwords = set(Dictionaries.get_stopwords())
        # log.info("Loading mention entity prior...")
        # Dictionaries.men_ent_prior_dict = Dictionaries.get_men_ent_prior_dict()
        # log.info("Loading processed items...")
        # Dictionaries.processed_items = Dictionaries.get_processed_items()
        # log.info("Loading processed entity...")
        # Dictionaries.processed_entities = Dictionaries.get_processed_entities()
        # log.info("loading spacy en web core md model..")
        # Dictionaries.spacy_init = spacy.load('en_core_web_md')
        # log.info("loading mention doc mappings...")
        # Dictionaries.id_doc_dict, Dictionaries.doc_men_dict = get_id_men_doc_dict()
        # log.info("loading maximum ent priors for every documents")
        # Dictionaries.max_ent_priors_dict = get_max_entity_prior()
        # Dictionaries.men_keyphrase_dict = Dictionaries.get_mention_keyphrase_dict()
        # log.info("Loading the total ent counts from Redis...")
        # Dictionaries.ent_total_counts = Dictionaries.redis_db_obj.\
        #     get_total_ents_counts_aida()

    @staticmethod
    def init_max_ent_priors():
        log.info("loading mention doc mappings...")
        Dictionaries.id_doc_dict, Dictionaries.doc_men_dict = \
            get_id_men_doc_dict()
        log.info("loading maximum ent priors for every documents")
        Dictionaries.max_ent_priors_dict_emnlp17 = get_max_entity_prior_emnlp17()
        Dictionaries.max_ent_priors_dict_2014 = get_max_entity_prior_2014()
        log.info("Loading doc mentions beg postions...")
        Dictionaries.doc_men_beg_pos_dict = get_doc_mentions_beg_pos_dict()

    @staticmethod
    def get_entity_keyphrase(entity):
        keyphrases = []
        id_men_ent_keyphrase_tuples = Dictionaries.db_conn_obj.\
            fetch_entity_keyphrases(entity)
        for subject_str, rela_str, object_str in id_men_ent_keyphrase_tuples:
            keyphrases.append([subject_str, rela_str, object_str])
        return keyphrases

    @staticmethod
    def doc_to_hash(doc_text):
        hash_alg = hashlib.md5()
        hash_alg.update(doc_text.encode())
        return hash_alg.hexdigest()

    @staticmethod
    def set_experiment_id(exp_id):
        Dictionaries.experiment_id = exp_id

    @staticmethod
    @abc.abstractmethod
    def get_processed_items():
        processed_items = []
        try:
            with open(os.path.join(result_path, 'ctx-resutlt.conll.csv'), 'r') as f:
                reader = csv.reader(f, delimiter=',')
                for row in reader:
                    processed_items.append(row[0])
        except IOError as e:
            log.exception("No processed cases!")
        return processed_items

    @staticmethod
    def get_stopwords():
        with open(os.path.join(os.path.dirname(database.__path__[0]), 'stopwords_en.txt'), 'r') as f:
            stopwords = [word.strip() for word in f]
        return stopwords

    @staticmethod
    def get_men_ent_prior_dict():
        men_ent_prior_dict = defaultdict(float)
        aida_db = database.db_kongzi2.PostgreSQLAIDA()
        men_ent_prior_tuples = aida_db.fetch_men_ent_prior()
        for men, ent, prior in men_ent_prior_tuples:
            men_ent_prior_dict[(men, ent)] = prior
        aida_db.conn.close()
        return men_ent_prior_dict

    @staticmethod
    def get_processed_entities():
        processed_entity = Dictionaries.db_conn_obj.fetch_processed_entities()

        return [entity[0] for entity in processed_entity]

    @staticmethod
    @abc.abstractmethod
    def get_mention_keyphrase_dict():
        with open(os.path.join(result_path, 'no_mention_keyphrase.conll'), 'a') as f:
            id_men_ent_keyphrase_dict = defaultdict(list)
            id_text = Dictionaries.db_conn_obj.fetch_id_text_conll_testb()
            for doc_id, doc_text in id_text:
                id_men_ent_keyphrase_tuples = Dictionaries.db_conn_obj.fetch_m_keyphrase_by_id(doc_to_hash(doc_text))
                if len(id_men_ent_keyphrase_tuples) == 0:
                    f.write(str(doc_id) + '\n')
                    continue
                for subject_str, rela_str, object_str in id_men_ent_keyphrase_tuples:
                    id_men_ent_keyphrase_dict[doc_id].append([subject_str, rela_str, object_str])
                id_men_ent_keyphrase_dict[doc_id] = json.dumps(id_men_ent_keyphrase_dict[doc_id])
        return id_men_ent_keyphrase_dict


def get_doc_mentions_beg_pos_dict():
    doc_mentions_tuples = Dictionaries.db_conn_obj.\
        fetch_doc_mens_beg_pos_source(Dictionaries.dataset_source)
    doc_id_mention_beg_pos_dict = defaultdict(dict)

    for doc_id, surfaceform, begoffset in doc_mentions_tuples:
        if doc_id not in doc_id_mention_beg_pos_dict:
            doc_id_mention_beg_pos_dict[doc_id] = defaultdict(list)

        doc_id_mention_beg_pos_dict[doc_id][surfaceform].append(begoffset)

    return doc_id_mention_beg_pos_dict


def get_id_men_doc_dict():
    id_men_doc_list = Dictionaries.db_conn_obj.fetch_id_men_doc_source(
        Dictionaries.dataset_source)
    id_doc_dict = dict([(x[0], x[2]) for x in id_men_doc_list])
    doc_men_dict = defaultdict(list)
    for _, men, doc in id_men_doc_list:
        doc_men_dict[doc].append(men)
    return id_doc_dict, doc_men_dict


def get_max_entity_prior():
    doc_ent_priors_dict = {}

    doc_ids = set([Dictionaries.id_doc_dict[x] for x in
                   range(Dictionaries.m_start, Dictionaries.m_end + 1)
                   if x in Dictionaries.id_doc_dict])
    for doc_id in doc_ids:
        mens = Dictionaries.doc_men_dict[doc_id]
        ent_priors_dict = defaultdict(float)
        normalized_mens = tuple(set([normalize_form(x) for x in set(mens)]))
        ent_priors = Dictionaries.prior_db_conn_obj.\
            fetch_entity_priors_by_mentions(normalized_mens)
        for ent, prior in ent_priors:
            ent_priors_dict[ent] = max(ent_priors_dict[ent], prior)
        doc_ent_priors_dict[doc_id] = ent_priors_dict
    return doc_ent_priors_dict


def get_max_entity_prior_emnlp17(batch_mode=True):
    doc_ent_priors_dict = {}

    doc_ids = set([Dictionaries.id_doc_dict[x] for x in
                   range(Dictionaries.m_start, Dictionaries.m_end + 1)
                   if x in Dictionaries.id_doc_dict])
    for doc_id in doc_ids:
        mens = Dictionaries.doc_men_dict[doc_id]
        ent_priors_dict = defaultdict(float)
        mens_set = set(mens)
        for men in mens_set:
            n_men, ent_priors = preprocess_mention(men)
            if batch_mode:
                if ent_priors:
                    ent_reds = get_redirect_entity_batch([ent for ent, prior in ent_priors])
                    for (ent, prior), ent_red in zip(ent_priors, ent_reds):
                        ent_priors_dict[ent_red] = max(ent_priors_dict[ent_red], prior)
            else:
                for ent, prior in ent_priors:
                    ent_red = get_redirect_entity(ent)
                    ent_priors_dict[ent_red] = max(ent_priors_dict[ent_red], prior)
        doc_ent_priors_dict[doc_id] = ent_priors_dict
    return doc_ent_priors_dict


def get_max_entity_prior_2014():
    doc_ent_priors_dict = {}

    doc_ids = set([Dictionaries.id_doc_dict[x] for x in
                   range(Dictionaries.m_start, Dictionaries.m_end + 1)
                   if x in Dictionaries.id_doc_dict])
    for doc_id in doc_ids:
        mens = Dictionaries.doc_men_dict[doc_id]
        ent_priors_dict = defaultdict(float)
        mens_set = set(mens)
        for men in mens_set:
            men_ent_priors_dict = get_men_ent_priors(men)
            for ent, prior in men_ent_priors_dict.items():
                ent_priors_dict[ent] = max(ent_priors_dict[ent], prior)
        doc_ent_priors_dict[doc_id] = ent_priors_dict
    return doc_ent_priors_dict


def get_redirect_entity(entity):
    wiki_id = 'en.wikipedia.org/wiki/' + entity
    wiki_id_re = Dictionaries.prior_db_conn_obj.fetch_ent_redirects(wiki_id)
    wiki_id_re = wiki_id_re[0].replace("en.wikipedia.org/wiki/", '') if wiki_id_re else entity
    return wiki_id_re


def get_redirect_entity_batch(entities):
    wiki_ids = ['en.wikipedia.org/wiki/' + ent for ent in entities]
    wiki_ids_re = dict(Dictionaries.prior_db_conn_obj.fetch_ent_redirects_batch(wiki_ids))

    results = []
    for ent, wiki_id in zip(entities, wiki_ids):
        wiki_id_re = wiki_ids_re.get(wiki_id)
        if wiki_id_re is not None:
            results.append(wiki_id_re.replace("en.wikipedia.org/wiki/", ''))
        else:
            results.append(ent)
    return results


def get_men_ent_priors(mention):
    men = mention.lower()
    ent_priors = Dictionaries.prior_db_conn_obj.fetch_entity_by_mention_2014(men)
    return dict(ent_priors)


def get_entity_counts(entity):
    wiki_id = 'en.wikipedia.org/wiki/' + entity
    count = Dictionaries.prior_db_conn_obj.fetch_ent_counts(wiki_id)
    count = count[0] if count else 0
    return count


def split_in_words(inputstr):
    pattern = re.compile(r'(\w+)')
    words = [word for word in re.findall(pattern, inputstr)]
    return words


def first_letter_to_uppercase(s):
    return s[:1].upper() + s[1:]


def modify_uppercase_phrase(s):
    if s == s.upper():
        words = split_in_words(s.lower())
        res = [first_letter_to_uppercase(w) for w in words]
        return ' '.join(res)
    else:
        return s


def preprocess_mention(m, apply_fix=True, candidate_filter=None):
    """
    Get normalized mentions and all its candidate entities.
    :param m:
    :param can_size
    :return:
    """
    cur_m = modify_uppercase_phrase(m)

    can_ents_cur_m = Dictionaries.prior_db_conn_obj.\
        fetch_entity_by_mention_size_emnlp17(cur_m, size=0)
    if cur_m == m:
        can_ents_m = can_ents_cur_m
    else:
        can_ents_m = Dictionaries.prior_db_conn_obj.fetch_entity_by_mention_size_emnlp17(m, size=0)

    mention_total_freq_m = 0 if not can_ents_m else len(can_ents_m)
    mention_total_freq_cur_m = 0 if not can_ents_cur_m else len(can_ents_cur_m)

    if not can_ents_cur_m:
        cur_m = m
        can_ents_cur_m = can_ents_m
    elif mention_total_freq_m and mention_total_freq_m > mention_total_freq_cur_m:
        cur_m = m
        can_ents_cur_m = can_ents_m

    # If we cannot find the exact mention in our index,
    # we try our luck to find it in a case insensitive index.
    if apply_fix:
        men_relaxed_form = cur_m.lower()
    else:
        men_relaxed_form = Dictionaries.redis_db_obj.get_men_relaxed_form_redis(cur_m.lower())

    if not can_ents_cur_m and men_relaxed_form:
        cur_m = men_relaxed_form
        can_ents_cur_m = Dictionaries.prior_db_conn_obj.\
            fetch_entity_by_mention_size_emnlp17(cur_m, size=0)

    if apply_fix:
        # XL: find from YAGO
        if not can_ents_cur_m:
            cur_m = normalize_form(m)
            can_ents_cur_m = Dictionaries.prior_db_conn_obj.fetch_entity_by_mention_size(cur_m, size=0)

    if candidate_filter is not None:
        #print('Candidate Filter: %s' % candidate_filter)
        can_ents_cur_m = [(ent, prior) for ent, prior in can_ents_cur_m if ent in candidate_filter]
    return cur_m, can_ents_cur_m


def escape_entity(entity):
    return str.encode(entity).decode('unicode-escape')


def get_all_entity_list():
    men_ent_dict = defaultdict(set)
    negative_entity_list_db = Dictionaries.db_conn_obj.fetch_all_entity_from_wiki_training()
    for mention, entity in negative_entity_list_db:
        men_ent_dict[mention].add(entity)
    return men_ent_dict


# def get_entity_context_dict_conll():
#     ent_context_dict = defaultdict(str)
#     entity_lists_db = Dictionaries.db_conn_obj.fetch_all_entity_from_conll_testb()
#     entity_lists = [escape_entity(entity[0]) for entity in entity_lists_db]
#     for entity in entity_lists:
#         ent_context_dict[entity] = get_entity_context(entity)
#     return ent_context_dict


# def get_entity_context_dict():
#     ent_context_dict = defaultdict(str)
#     entity_lists_db = Dictionaries.db_conn_obj.fetch_all_entity_from_processed_clauses()
#     entity_lists = [entity[0] for entity in entity_lists_db]
#     for entity in entity_lists:
#         ent_context_dict[entity] = get_entity_context(entity)
#
#     return ent_context_dict


def pre_process_wikipage_tac_kbp2014(wiki_text):
    # todo pre-process the Wikipedia article, some short paragraphs titles need to be filtered
    # for _ in range(5):
    # wiki_text = re.sub('(\n\n|^).*(\n(\n|$))', '\n\n', wiki_text)
    # print(wiki_text)
    wiki_text = wiki_text.replace('-\n', '-').replace('\n\n', '-----newpar-----').replace('\n', ' ').replace(
        '-----newpar-----', '.\n\n')
    # wiki_text = re.sub('\n.', '. ', wiki_text)
    return wiki_text.strip()


def get_mention_context_dict():
    id_men_ent_context_dict = defaultdict(str)
    id_men_ent_context_tuples = Dictionaries.db_conn_obj.fetch_mention_entity_pairs_training()
    for doc_id, men, ent, context in id_men_ent_context_tuples:
        id_men_ent_context_dict[doc_id] = context
    return id_men_ent_context_dict


def get_mention_context_dict_conll():
    id_men_ent_context_dict = defaultdict(str)
    id_men_ent_context_tuples = Dictionaries.db_conn_obj.fetch_mention_entity_pairs_conll_testb()
    for doc_id, men, ent, context in id_men_ent_context_tuples:
        id_men_ent_context_dict[doc_id] = context
    return id_men_ent_context_dict


def doc_to_hash(doc_text):
    hash_alg = hashlib.md5()
    hash_alg.update(doc_text.encode())
    return hash_alg.hexdigest()


def normalize_form(mention):
    return mention if len(mention) < 4 else str(mention).upper()


def get_entity_keyphrase_dict():
    id_men_ent_keyphrase_dict = defaultdict(list)
    id_men_ent_keyphrase_tuples = Dictionaries.db_conn_obj.fetch_entity_keyphrases_all()
    for doc_id, subject_str, rela_str, object_str in id_men_ent_keyphrase_tuples:
        id_men_ent_keyphrase_dict[doc_id].append((subject_str, rela_str, object_str))
    return id_men_ent_keyphrase_dict


# def get_entity_context(entity):
#     try:
#         wiki_content = Dictionaries.db_conn_obj.fetch_entity_context_from_training(entity)
#         if wiki_content is None or wiki_content[0] == '':
#             wiki_content = Dictionaries.db_conn_obj.fetch_entity_context_from_online(entity)
#             if wiki_content is None or wiki_content[0] == '':
#                 wiki_content = get_training_article(entity)
#                 if wiki_content != '':
#                     try:
#                         print("saving article: " + entity)
#                         Dictionaries.db_conn_obj.save_training_wiki_article(
#                             [entity, "https://en.wikipedia.org/wiki/" + entity, wiki_content, len(wiki_content)])
#                         print("Finished saving article: " + entity)
#                     except:
#                         traceback.print_exc()
#                         # sys.exit(1)
#                 else:
#                     wiki_content = Dictionaries.db_conn_obj.fetch_entity_context_from_tackbp(entity)
#                     if wiki_content is not None:
#                         wiki_content = pre_process_wikipage_tac_kbp2014(wiki_content[0])
#                     else:
#                         wiki_content = ''
#             else:
#                 wiki_content = wiki_content[0]
#                 wiki_content = re.sub(r'((=){2,}[\w| |–|:]+(=){2,})', '', wiki_content)
#                 wiki_content = re.sub(r'(/.*[\u0250-\u02FF].*/)', '', wiki_content)
#                 wiki_content = wiki_content.replace("\n\n\n", "\n")
#         else:
#             wiki_content = wiki_content[0]
#     except:
#         traceback.print_exc()
#         wiki_content = ''
#     return wiki_content.strip()


# def get_training_article(entity):
#     result = ''
#     header = {'User-Agent': 'Mozilla/5.0'}  # Needed to prevent 403 error on Wikipedia
#     wiki_url = "https://en.wikipedia.org/wiki/"
#     respond = requests.get(wiki_url + entity, headers=header)
#     soup = BeautifulSoup(respond.text)
#     for paragraph in soup.find_all("p"):
#         res_text = re.sub(r'\[([0-9]+)\]', '', paragraph.text)
#         result += res_text + '\n'
#     return result

if __name__ == '__main__':
    pass
    # get_entity_context_dict_conll()
    # get_entity_context('Basarab_Panduru')
