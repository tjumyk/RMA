import task_scheduler
import os
from collections import defaultdict
import json

import task_scheduler.database.db_psql
import task_scheduler.schelduler
import task_scheduler.redis_db.redis_init
from task_scheduler import logging_config
import task_scheduler.database.db_kongzi2
# import nerd.disambiguation.candidates

log = logging_config.getLogger()


class Dictionaries:
    config_path = None
    db = None
    prior_db_conn_obj = None
    tasks_done = []
    candidate_ents = []
    seed_entity = None
    processed_entity = []
    processed_mention = []
    hop = []
    redis_db_obj = task_scheduler.redis_db.redis_init.RedisConn()


def init_dictionaries():
    log.info("Loading configuration...")
    Dictionaries.config_path = os.path.join(os.path.dirname(task_scheduler.__path__[0]), 'config', 'config.json')
    Dictionaries.db = task_scheduler.database.db_psql.PostgreSQLDB()
    Dictionaries.prior_db_conn_obj = task_scheduler.database.db_kongzi2.PostgreSQLAIDA()
    # Dictionaries.processed_mention = get_processed_mention()
    # with open(Dictionaries.config_path, 'r') as f:
    #     config = json.load(f)
    # Dictionaries.seed_entity = config['seed_entity']
    # Dictionaries.hop = config['hop']
    # Dictionaries.clausie_url = config['clausie']['host']
    # Dictionaries.tasks_done = task_scheduler.schelduler.get_tasks()
    log.info("Finished loading configuration.")
    # Dictionaries.word_idf_dict = db.get_all_words_idf_dict()
    # Dictionaries.word_mi_dict = db.get_all_words_mi_dict()
    # Dictionaries.collection_size = db.get_collection_size()
    # db.keyphrase_db.close_db_conn()
    # Dictionaries.stopwords = set(get_stopwords())


# def get_stopwords():
#     with open(os.path.join(nerd.__path__[0], 'stopwords_en.txt'), 'r') as f:
#         stopwords = [word.strip() for word in f]
#     return stopwords

def get_processed_mention():
    mentions = Dictionaries.db.fetch_processed_mentions()
    return [mention[0] for mention in mentions]

if __name__ == '__main__':
    init_dictionaries()
    get_processed_mention()