import redis_db
import json
import redis
import os


class RedisConn(object):
    def __init__(self):
        config_path = redis_db.__path__[0]
        config_path = os.path.join(config_path, 'config.json')
        # log.info('config.path: %s' % config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        db = config['redis_db']

        self.conn = redis.Redis(password=db['password'], host=db['host'], port=db['port'], decode_responses=True)

    def get_collection_size(self):
        collection_size = self.conn.get('collection_size')
        return float(collection_size)

    # def get_men_kps_redis(self, redis_key, q_id):
    #     kps_tmp = []
    #     if self.conn.hexists('men_kps_conll', q_id):
    #         kps_tmp = json.loads(self.conn.hmget('men_kps_conll', q_id)[0])
    #     return kps_tmp

    def is_processed_items_redis(self, exp_id, d_t, id_ent):
        return self.conn.sismember('processed_queries:::::' + exp_id + ":::::" + d_t, id_ent)

    def is_processed_ents_redis(self, ent):
        return self.conn.sismember('processed_ents', ent)

    def is_existed_redis(self, redis_key):
        return self.conn.exists(redis_key)

    def add_processed_items_redis(self, id_ent):
        return self.conn.sadd('processed_queries', id_ent)

    def save_set_redis(self, redis_key, set_val):
        if isinstance(set_val, (list, tuple, set)):
            return self.conn.sadd(redis_key, *set_val)
        return self.conn.sadd(redis_key, set_val)

    def get_diff_sets_redis(self, l_set, r_set_key):
        r_p = self.conn.pipeline()
        if l_set:
            r_p.sadd('l_set_tmp', *l_set)
            r_p.sdiff('l_set_tmp', r_set_key)
            r_p.delete('l_set_tmp')
            res = r_p.execute()
            return res[1]
        else:
            return None

    def save_list_redis(self, redis_key, list_val):
        if isinstance(list_val, (list, tuple, set)):
            return self.conn.rpush(redis_key, *list_val)
        return self.conn.rpush(redis_key, list_val)

    def get_men_kps_redis(self, redis_key, men_id):
        kps = self.conn.hmget(redis_key, men_id)[0]
        return json.loads(kps) if kps else []

    def get_ent_ctx_redis(self, redis_key, ent_id):
        kps = self.conn.hmget(redis_key, ent_id)[0]
        return json.loads(kps) if kps else []

    def save_ent_ctx_redis(self, redis_key, ent_id, ctx):
        if not self.conn.hexists(redis_key, ent_id):
            self.conn.hmset(redis_key, {ent_id: json.dumps(ctx)})

    def get_word_weight_redis(self, redis_key, word):
        weight = self.conn.hmget(redis_key, word)[0]
        return float(weight) if weight else weight

    def save_dict_redis(self, redis_key, *dict_val):
        if len(dict_val) == 2:
            return self.conn.hset(redis_key, dict_val[0], dict_val[1])
        if len(dict_val) == 1 and isinstance(dict_val[0], dict):
            return self.conn.hmset(redis_key, dict_val[0])
        else:
            raise Exception

    def get_ents_counts_aida(self, ent):
        count = self.conn.hmget('entity_counts_aida', ent)[0]
        return float(count) if count else 0

    def get_total_ents_counts_aida(self):
        count = self.conn.get('entity_counts_total_aida')
        return float(count) if count else 0

    def get_ent_article_redis(self, redis_key, ent_id):
        article = self.conn.hmget(redis_key, ent_id)[0]
        return article if article else ''

    def get_ent_tokens_redis(self, ent_id):
        tokens = self.conn.hmget('ent-tok-ner-spacy', ent_id)[0]
        return json.loads(tokens) if tokens else {}

    def get_men_tokens_redis(self, doc_id):
        tokens = self.conn.hmget('men-tok-ner-spacy-conll', doc_id)[0]
        return json.loads(tokens) if tokens else {}
