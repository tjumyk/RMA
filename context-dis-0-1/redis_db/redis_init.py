import redis_db
import json
import redis
import os
import socket

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

    def is_processed_items_redis(self, exp_id, d_s, d_t, id_ent):
        return self.conn.sismember('processed_queries:::::' + exp_id + ":::::" + d_s + ":::::" + d_t, id_ent)

    def del_n_can_redis(self, exp_id, d_s, d_t):
        return self.conn.delete('no_candidate:::::' + exp_id + d_s + ":::::" + ":::::" + d_t)

    def del_missed_g_can_redis(self, exp_id, d_s, d_t, c_size=50):
        return self.conn.delete('missed_g_candidate:::::' + exp_id + ":::::" + d_s + ":::::" + d_t + ":::::" +
                                str(c_size))

    def del_n_g_info_can_redis(self, exp_id, d_s, d_t):
        return self.conn.delete('no_g_candidate_info:::::' + exp_id + ":::::" + d_s + ":::::" + d_t)

    def is_processed_ents_redis(self, ent_repo_id, ent):
        return self.conn.hexists(ent_repo_id, ent)

    def is_existed_redis(self, redis_key):
        return self.conn.exists(redis_key)

    def add_processed_items_redis(self, id_ent):
        return self.conn.sadd('processed_queries', id_ent)

    def save_set_redis(self, redis_key, set_val):
        if isinstance(set_val, (list, tuple, set)):
            return self.conn.sadd(redis_key, *set_val)
        return self.conn.sadd(redis_key, set_val)

    def get_n_processed_ents_redis(self, ent_repo_id, l_set):
        p_ents = self.conn.hkeys(ent_repo_id)
        diff_sets = set(l_set) - set(p_ents)
        return diff_sets

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

    def get_ents_counts_emnlp17(self, ent):
        count = self.conn.hmget('entity_counts_emnlp17', ent)[0]
        return float(count) if count else 0

    def get_total_ents_counts_aida(self):
        count = self.conn.get('entity_counts_total_aida')
        return float(count) if count else 0

    def get_ent_tokens_redis(self, ent_id, redis_key='ent-tok-ner-spacy_md'):
        tokens = self.conn.hmget(redis_key, ent_id)[0]
        return json.loads(tokens) if tokens else {}

    def get_men_tokens_redis(self, data_source, doc_id, redis_key='men-tok-ner-spacy_md-'):
        tokens = self.conn.hmget(redis_key + data_source, doc_id)[0]
        return json.loads(tokens) if tokens else {}

    def get_men_ents_tokens_idf_redis(self, men, redis_key=''):
        ent_tok_dict = self.conn.hmget('idf:::50' + redis_key, men)[0]
        return json.loads(ent_tok_dict) if ent_tok_dict else {}

    def get_men_ents_tokens_entropy_redis(self, men, redis_key=''):
        ent_tok_dict = self.conn.hmget('entropy:::50' + redis_key, men)[0]
        return json.loads(ent_tok_dict) if ent_tok_dict else {}

    def get_men_ents_tokens_tf_redis(self, men, redis_key=''):
        ent_tok_dict = self.conn.hmget('tf:::50' + redis_key, men)[0]
        return json.loads(ent_tok_dict) if ent_tok_dict else {}

    def save_men_ents_tokens_idf_redis(self, men, ent_idf_dict, redis_key=''):
        return self.conn.hset('idf:::50' + redis_key, men, json.dumps(ent_idf_dict))

    def save_men_ents_tokens_entropy_redis(self, men, ent_entropy_dict, redis_key=''):
        return self.conn.hset('entropy:::50' + redis_key, men, json.dumps(ent_entropy_dict))

    def save_men_ents_tokens_tf_redis(self, men, ent_tf_dict, redis_key=''):
        return self.conn.hset('tf:::50' + redis_key, men, json.dumps(ent_tf_dict))

    def get_ents_stfd_by_doc(self, doc_id, data_source='aida_conll'):
        ents = self.conn.hmget('men-ner-stanford-' + data_source, doc_id)[0]
        return json.loads(ents) if ents else {}

    def missed_g_ent(self, q_id, data_source='aida_conll', can_size=50, data_type='testb'):
        return self.conn.sismember(
            'missed_g_candidate:::::' + 'basic_feature_coref_smooth' + ":::::" + data_source + ":::::" + data_type +
            ":::::" + str(can_size), str(q_id))

    def missed_g_ent_emnlp17(self, q_id, data_source='aida_conll', can_size=50, data_type='testb'):
        return self.conn.sismember(
            'missed_g_candidate:::::' + 'basic_context_emnlp17_spacy_sm_50_new' + ":::::" + data_source + ":::::" + data_type +
            ":::::" + str(can_size), str(q_id))

    def fetch_inlinks_redis(self, ent, link_type='inlinks'):
        inlinks = self.conn.hmget(link_type, ent)[0]
        return json.loads(inlinks) if inlinks else []

    def has_inlinks_redis(self, ent, link_type='inlinks'):
        inlinks = self.conn.hexists(link_type, ent)
        return inlinks

    def save_inlinks_redis(self, ent, inlinks, link_type='inlinks'):
        self.conn.hset(link_type, ent, json.dumps(inlinks))

    def fetch_outlinks_redis(self, ent, link_type='outlinks'):
        outlinks = self.conn.hmget(link_type, ent)[0]
        return json.loads(outlinks) if outlinks else []

    def has_outlinks_redis(self, ent, link_type='outlinks'):
        inlinks = self.conn.hexists(link_type, ent)
        return inlinks

    def save_outlinks_redis(self, ent, outlinks, link_type='outlinks'):
        self.conn.hset(link_type, ent, json.dumps(outlinks))

    def get_id_ents_by_doc_redis(self, data_source, data_type, doc_id):
        suffix = socket.gethostname()
        id_ents = self.conn.hmget('doc-predicted-ents:%s:%s:%s' % (data_source, data_type, suffix), doc_id)[0]
        return json.loads(id_ents) if id_ents else []

    def get_id_ents_extended_by_doc_redis(self, data_source, data_type, doc_id):
        suffix = socket.gethostname()
        id_ents = self.conn.hmget('doc-predicted-ents-ext:%s:%s:%s' % (data_source, data_type, suffix), doc_id)[0]
        return json.loads(id_ents) if id_ents else []

    def get_men_relaxed_form_redis(self, men):
        relaxed_men = self.conn.hmget('mention_lower_to_one_upper', men)[0]
        return relaxed_men

    def get_redirected_ent_redis(self, ent):
        relaxed_men = self.conn.hmget('ent_red_map', ent)[0]
        return relaxed_men if relaxed_men else ent

    def get_ent_pair_info(self, entity, coh_ent):
        if entity > coh_ent:
            entity, coh_ent = coh_ent, entity
        key = str((entity, coh_ent))
        value = self.conn.hget('ent_pair_info', key)
        return json.loads(value) if value else None

    def get_ent_pair_info_ext(self, entity, coh_ent):
        if entity > coh_ent:
            entity, coh_ent = coh_ent, entity
        key = str((entity, coh_ent))
        value = self.conn.hget('ent_pair_info_ext', key)
        return json.loads(value) if value else None

    def save_ent_pair_info(self, entity, coh_ent, pair_info):
        if entity > coh_ent:
            entity, coh_ent = coh_ent, entity
        key = str((entity, coh_ent))
        self.conn.hset('ent_pair_info', key, json.dumps(pair_info))

    def save_ent_pair_info_ext(self, entity, coh_ent, pair_info_ext):
        if entity > coh_ent:
            entity, coh_ent = coh_ent, entity
        key = str((entity, coh_ent))
        self.conn.hset('ent_pair_info_ext', key, json.dumps(pair_info_ext))

