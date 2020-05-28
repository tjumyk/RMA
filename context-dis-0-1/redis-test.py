import database.db_kongzi3
import database.db_kongzi2
from redis_db.redis_init import RedisConn
from const_vars.constant_conll_testb import Dict_Co
from itertools import islice


def chunks(data, SIZE=10000):
    it = iter(data)
    for i in range(0, len(data), SIZE):
        yield {k: data[k] for k in islice(it, SIZE)}


def load_word_idf_redis():
    r_conn = RedisConn().conn
    word_idf_dict = dict(database.db_kongzi3.PostgreSQL().fetch_all_words_idf())
    print(type(word_idf_dict))
    # res = r_conn.hmget('word-idf', 'a')
    for item in chunks(word_idf_dict, 10000):
        res = r_conn.hmset('idf', item)
        print(res)


def load_word_ent_mi_redis():
    r_conn = RedisConn().conn
    word_tuples = database.db_kongzi3.PostgreSQL().fetch_all_words_mi()
    word_mi_dict = {(a, b): c for a, b, c in word_tuples}
    # word_mi_dict = {('a', 'b'): 123}
    for item in chunks(word_mi_dict, 50000):
        res = r_conn.hmset('mi', item)
        print(res)


def load_collection_size_redis():
    r_conn = RedisConn().conn
    word_tuples = database.db_kongzi3.PostgreSQL().fetch_collection_size()[0][0]
    res = r_conn.set('collection_size', word_tuples)
    print(res)


def load_all_redis():
    # test_case = {"a": json.dumps([[1,2,3], ['2', 4, '5'], ['4, 6, 7']]), 'b':json.dumps([[1,4]])}
    # test_case = json.dumps(test_case)
    r_conn = RedisConn().conn
    # res = r_conn.hmset('test-ctx', test_case)
    # res = r_conn.hmget('test-ctx', 'a')
    # print(type(res))
    # print(res)
    ss = r_conn.hmset('entity_counts_aida', Dict_Co.processed_entities)
    # ss = r_conn.sadd('processed_ents', *Dict_Co.processed_entities)
    # ss = r_conn.hmset('men_kps_conll_train', Dict_Co.men_keyphrase_dict)
    # r_conn.set('collection_size', 67780)
    # ss = r_conn.sadd('processed_queries', *Dict_Co.processed_items)
    print(ss)


def Load_entity_articles():
    r_conn = RedisConn().conn
    ent_tuples = database.db_kongzi3.PostgreSQL().fetch_all_entities_context_from_training()
    ent_dict = dict(ent_tuples)
    for item in chunks(ent_dict, 50000):
        res = r_conn.hmset('entity_article_training', item)
        print(res)


def load_entity_counts():
    r_conn = RedisConn().conn
    ent_path = './candidate_ents_conll_emnlp17_50.csv'
    # ent_path = './candidate_ents_wiki_uiuc.csv'
    # ents = []
    with open(ent_path, 'r', encoding='utf-8') as f:
        ents = [x.strip('\n') for x in f.readlines()]
    ent_tuples = []
    ents = list(set(ents))
    for ent in ents:
        print("Processing entity {}".format(ent))
        count = database.db_kongzi2.PostgreSQLAIDA().fetch_ents_count_emnlp17(ent)[0]
        count = count[0] if count else 0
        ent_tuples.append((ent, count))
    ent_dict = dict(ent_tuples)
    # word_mi_dict = {('a', 'b'): 123}
    for item in chunks(ent_dict, 10000):
        res = r_conn.hmset('entity_counts_emnlp17', item)
        print(res)


if __name__ == '__main__':
    # s = r_conn.hmget('user:1', 'name')
    # print(type(s))i
    # load_word_idf_redis()
    # load_word_ent_mi_redis()
    # load_collection_size_redis()
    # test_set_redis()
    # Dict_Co.init_dictionaries()
    # load_all_redis()
    load_entity_counts()
    # Load_entity_articles()
