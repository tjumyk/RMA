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
from bs4 import BeautifulSoup

result_path = result_folder.__path__[0]

log = logging_config.get_logger()


class Dictionaries(object):
    __metaclass__ = abc.ABCMeta

    collection_size = 0
    threads = 4
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
    experiment_id = ''
    ent_total_counts = 0

    @staticmethod
    def init_dictionaries():
        log.info("Loading words IDF...")
        # Dictionaries.word_idf_dict = Dictionaries.db_conn_obj.get_all_words_idf_dict()
        log.info("Loading entity words MI...")
        # Dictionaries.word_mi_dict = Dictionaries.db_conn_obj.get_all_words_mi_dict()
        log.info("Loading collection size...")
        # Dictionaries.collection_size = Dictionaries.redis_db_obj.get_collection_size()
        log.info("Loading Stopwords...")
        # Dictionaries.stopwords = set(Dictionaries.get_stopwords())
        log.info("Loading mention entity prior...")
        # Dictionaries.men_ent_prior_dict = Dictionaries.get_men_ent_prior_dict()
        log.info("Loading processed items...")
        # Dictionaries.processed_items = Dictionaries.get_processed_items()
        log.info("Loading processed entity...")
        # Dictionaries.processed_entities = Dictionaries.get_processed_entities()
        log.info("loading spacy en model..")
        Dictionaries.spacy_init = spacy.load('en_core_web_sm')
        log.info("loading mention keyphrase...")
        # Dictionaries.men_keyphrase_dict = Dictionaries.get_mention_keyphrase_dict()
        log.info("Loading the total ent counts from Redis...")
        # Dictionaries.ent_total_counts = Dictionaries.redis_db_obj.get_total_ents_counts_aida()

    @staticmethod
    def get_entity_keyphrase(entity):
        keyphrases = []
        id_men_ent_keyphrase_tuples = Dictionaries.db_conn_obj.fetch_entity_keyphrases(entity)
        for subject_str, rela_str, object_str in id_men_ent_keyphrase_tuples:
            keyphrases.append([subject_str, rela_str, object_str])
        return keyphrases

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


def escape_entity(entity):
    return str.encode(entity).decode('unicode-escape')


def get_all_entity_list():
    men_ent_dict = defaultdict(set)
    negative_entity_list_db = Dictionaries.db_conn_obj.fetch_all_entity_from_wiki_training()
    for mention, entity in negative_entity_list_db:
        men_ent_dict[mention].add(entity)
    return men_ent_dict


def get_entity_context_dict_conll():
    ent_context_dict = defaultdict(str)
    entity_lists_db = Dictionaries.db_conn_obj.fetch_all_entity_from_conll_testb()
    entity_lists = [escape_entity(entity[0]) for entity in entity_lists_db]
    for entity in entity_lists:
        ent_context_dict[entity] = get_entity_context(entity)
    return ent_context_dict


def get_entity_context_dict():
    ent_context_dict = defaultdict(str)
    entity_lists_db = Dictionaries.db_conn_obj.fetch_all_entity_from_processed_clauses()
    entity_lists = [entity[0] for entity in entity_lists_db]
    for entity in entity_lists:
        ent_context_dict[entity] = get_entity_context(entity)

    return ent_context_dict


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


def get_entity_keyphrase_dict():
    id_men_ent_keyphrase_dict = defaultdict(list)
    id_men_ent_keyphrase_tuples = Dictionaries.db_conn_obj.fetch_entity_keyphrases_all()
    for doc_id, subject_str, rela_str, object_str in id_men_ent_keyphrase_tuples:
        id_men_ent_keyphrase_dict[doc_id].append((subject_str, rela_str, object_str))
    return id_men_ent_keyphrase_dict


def get_entity_context(entity):
    try:
        wiki_content = Dictionaries.db_conn_obj.fetch_entity_context_from_training(entity)
        if wiki_content is None or wiki_content[0] == '':
            wiki_content = Dictionaries.db_conn_obj.fetch_entity_context_from_online(entity)
            if wiki_content is None or wiki_content[0] == '':
                wiki_content = get_training_article(entity)
                if wiki_content != '':
                    try:
                        print("saving article: " + entity)
                        Dictionaries.db_conn_obj.save_training_wiki_article(
                            [entity, "https://en.wikipedia.org/wiki/" + entity, wiki_content, len(wiki_content)])
                        print("Finished saving article: " + entity)
                    except:
                        traceback.print_exc()
                        # sys.exit(1)
                else:
                    wiki_content = Dictionaries.db_conn_obj.fetch_entity_context_from_tackbp(entity)
                    if wiki_content is not None:
                        wiki_content = pre_process_wikipage_tac_kbp2014(wiki_content[0])
                    else:
                        wiki_content = ''
            else:
                wiki_content = wiki_content[0]
                wiki_content = re.sub(r'((=){2,}[\w| |–|:]+(=){2,})', '', wiki_content)
                wiki_content = re.sub(r'(/.*[\u0250-\u02FF].*/)', '', wiki_content)
                wiki_content = wiki_content.replace("\n\n\n", "\n")
        else:
            wiki_content = wiki_content[0]
    except:
        traceback.print_exc()
        wiki_content = ''
    return wiki_content.strip()


def get_training_article(entity):
    result = ''
    header = {'User-Agent': 'Mozilla/5.0'}  # Needed to prevent 403 error on Wikipedia
    wiki_url = "https://en.wikipedia.org/wiki/"
    respond = requests.get(wiki_url + entity, headers=header)
    soup = BeautifulSoup(respond.text)
    for paragraph in soup.find_all("p"):
        res_text = re.sub(r'\[([0-9]+)\]', '', paragraph.text)
        result += res_text + '\n'
    return result

if __name__ == '__main__':
    get_entity_context_dict_conll()
    # get_entity_context('Basarab_Panduru')