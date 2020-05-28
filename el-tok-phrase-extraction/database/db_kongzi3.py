from typing import List

import psycopg2
import json
import database
import os

# config_path = database.__path__[0]

class PostgreSQL(object):
    def __init__(self):
        config_path = database.__path__[0]
        config_path = os.path.join(config_path, 'config.json')
        # log.info('config.path: %s' % config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        db = config['db']

        self.conn = psycopg2.connect(database=db['database'], user=db['user'], password=db['password'], host=db['host'],
                                     port=db['port'])

    def fetch_wikipage_by_title(self, wiki_title):
        cur = self.conn.cursor()
        sql = "SELECT entity_id,wiki_text FROM kb_tac2014 WHERE wiki_title=(E\'%s\')"
        cur.execute(sql % wiki_title.replace("'", "\\'"))
        row = cur.fetchone()
        # print(row)
        cur.close()
        return ('NIL', None) if row is None else row

    def fetch_wikipage_by_title_online(self, wiki_title):
        cur = self.conn.cursor()
        sql = "SELECT content FROM wiki_online WHERE entity_id=(E\'%s\')"
        cur.execute(sql % wiki_title.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        # print(row)
        return row

    def save_wiki_online(self, wiki_info):
        cur = self.conn.cursor()
        sql = "INSERT INTO wiki_online (entity_id, url, title, summary, content, category, section, reference, link, image) VALUES (" + ",".join(
            ["%s"] * len(wiki_info)) + ")"
        # print(sql)
        cur.execute(sql, wiki_info)
        self.conn.commit()
        cur.close()

    def save_training_cases(self, training):
        cur = self.conn.cursor()
        args_str = ','.join(cur.mogrify("(%s,%s,%s,%s,%s,%s,%s,%s, %s)", x).decode('utf-8') for x in training)
        cur.execute("INSERT INTO wiki_training (doc_id, surfaceform, length, begoffset, entity, doc_text, wiki_url, type, candidate_size_aida)VALUES " + args_str)
        self.conn.commit()
        cur.close()

    def fetch_training_mentions_kore(self):
        cur = self.conn.cursor()
        sql = 'select surfaceform, entity from kore'
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_inlink_dict(self, entity):
        cur = self.conn.cursor()
        sql = "SELECT inlink_dict.inlink_entity FROM inlink_dict WHERE entity=(E\'%s\');"
        cur.execute(sql % entity.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        return row

    def fetch_entity_keyphrases(self, entity):
        cur = self.conn.cursor()
        sql = "SELECT  subject_coref, relation_str, object_coref FROM wiki_processed_clauses WHERE entity_id = (E\'%s\');"
        cur.execute(sql % entity.replace("'", "\\'"))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_m_keyphrase_by_id(self, m_id):
        cur = self.conn.cursor()
        sql = "SELECT  subject_coref, relation_str, object_coref FROM input_processed_clauses WHERE doc_id = (E\'%s\');"
        cur.execute(sql % m_id.replace("'", "\\'"))
        rows = cur.fetchall()
        cur.close()
        return rows


    def fetch_id_text_kore(self):
        cur = self.conn.cursor()
        sql = "SELECT kore.id, kore.doc_text FROM kore WHERE kore.entity != 'NIL'"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_id_text_conll_testb(self):
        cur = self.conn.cursor()
        sql = "SELECT aida_conll.id, aida_conll.doc_text FROM aida_conll WHERE aida_conll.annotation != 'NIL' and type='train'"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_from_wiki_training(self, size):
        cur = self.conn.cursor()
        sql = "SELECT DISTINCT(surfaceform) FROM wiki_training WHERE candidate_size_aida>=%s;"
        cur.execute(sql % size)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_from_wiki_training_by_entity(self, entity, size):
        cur = self.conn.cursor()
        sql = "SELECT DISTINCT(surfaceform) FROM wiki_training WHERE wiki_training.entity=(E\'%s\') and candidate_size_aida>=%s;"
        cur.execute(sql % (entity.replace("'", "\\'"), size))
        rows = cur.fetchall()
        cur.close()
        return rows

    def save_training_wiki_article(self, wiki_info):
        cur = self.conn.cursor()
        sql = "INSERT INTO wiki_article_training (entity_id, url, content, length) VALUES (" + ",".join(
            ["%s"] * len(wiki_info)) + ")"
        # print(sql)
        cur.execute(sql, wiki_info)
        self.conn.commit()
        cur.close()

    # def save_training_wiki_article(self, wiki_info):
    #     cur = self.conn.cursor()
    #     sql = "INSERT INTO wiki_article_training (entity_id, url, content, length) VALUES (" + ",".join(
    #         ["%s"] * len(wiki_info)) + ")"
    #     # print(sql)
    #     cur.execute(sql, wiki_info)
    #     self.conn.commit()
    #     cur.close()

    def fetch_training_wiki_article(self, wiki_id):
        cur = self.conn.cursor()
        sql = "SELECT content FROM wiki_online WHERE entity_id=(E\'%s\')"
        cur.execute(sql % wiki_id.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        # print(row)
        return row

    def fetch_mention_entity_pairs_conll_testb(self, q_id):
        cur = self.conn.cursor()
        sql = "SELECT aida_conll.id, aida_conll.surfaceform, aida_conll.annotation, aida_conll.doc_text, aida_conll.doc_id FROM aida_conll WHERE aida_conll.id=%s AND annotation != 'NIL'"
        cur.execute(sql % q_id)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_entity_pairs_conll_testb_by_size(self, q_id):
        cur = self.conn.cursor()
        sql = "SELECT aida_conll.id, aida_conll.surfaceform, aida_conll.annotation, aida_conll.doc_text, aida_conll.doc_id FROM aida_conll WHERE aida_conll.id=%s AND annotation != 'NIL';"
        cur.execute(sql % q_id) 
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_entity_pairs_kore(self)-> List[tuple]:
        '''
        
        :return: List of Tuples
        '''
        cur = self.conn.cursor()
        sql = "SELECT kore.id, kore.surfaceform, kore.entity, kore.doc_text FROM kore WHERE kore.entity != 'NIL'"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_entity_pairs_training(self):
        cur = self.conn.cursor()
        sql = "SELECT wiki_training.id, wiki_training.surfaceform, wiki_training.entity, wiki_training.doc_text FROM wiki_training WHERE candidate_size_aida>9 AND wiki_training.id IN (SELECT doc_id FROM training_processed_clauses) AND wiki_training.entity IN (SELECT wiki_processed_clauses.entity_id FROM wiki_processed_clauses)"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_keyphrase_training(self):
        cur = self.conn.cursor()
        sql = "SELECT wiki_training.id, wiki_training.surfaceform, wiki_training.entity, wiki_training.doc_text FROM wiki_training WHERE candidate_size_aida>9 AND wiki_training.id IN (SELECT doc_id FROM training_processed_clauses) AND wiki_training.entity IN (SELECT wiki_processed_clauses.entity_id FROM wiki_processed_clauses)"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows


    def fetch_mention_entity_context(self, mention, entity):
        cur = self.conn.cursor()
        sql = "SELECT id, doc_text From wiki_training WHERE surfaceform=(E\'%s\') and entity=(E\'%s\')"
        cur.execute(sql % (mention.replace("'", "\\'"), entity.replace("'", "\\'")))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_context_by_id(self, query_id):
        cur = self.conn.cursor()
        sql = "SELECT doc_text From wiki_training WHERE wiki_training.id=%s"
        cur.execute(sql % query_id)
        rows = cur.fetchone()
        cur.close()
        return rows

    def fetch_mention_entity_keyphrases(self, doc_id):
        cur = self.conn.cursor()
        sql = "SELECT subject_coref, relation_str, object_coref From training_processed_clauses WHERE doc_id=%s"
        cur.execute(sql % doc_id)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mention_entity_keyphrases_all(self):
        cur = self.conn.cursor()
        sql = "SELECT doc_id, subject_coref, relation_str, object_coref From training_processed_clauses"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_entity_keyphrases_all(self):
        cur = self.conn.cursor()
        sql = "SELECT wiki_processed_clauses.entity_id, wiki_processed_clauses.subject_coref, wiki_processed_clauses.relation_str, wiki_processed_clauses.object_coref From wiki_processed_clauses WHERE wiki_processed_clauses.entity_id in (SELECT DISTINCT (entity) FROM wiki_training)"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_processed_entities(self):
        cur = self.conn.cursor()
        sql = "SELECT DISTINCT wiki_processed_clauses.entity_id From wiki_processed_clauses;"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_entity_context_from_training(self, entity):
        cur = self.conn.cursor()
        sql = "SELECT content FROM wiki_article_training WHERE entity_id=(E\'%s\')"
        cur.execute(sql % entity.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        # print(row)
        return row

    def fetch_entities_context_from_training(self, entities):
        cur = self.conn.cursor()
        sql = "SELECT entity_id, content FROM wiki_article_training WHERE entity_id=(E\'%s\')"
        cur.execute(sql % entities.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        # print(row)
        return row

    def fetch_entity_context_from_online(self, wiki_title):
        cur = self.conn.cursor()
        sql = "SELECT content FROM wiki_online WHERE entity_id=(E\'%s\')"
        cur.execute(sql % wiki_title.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        # print(row)
        return row

    def fetch_entities_context_from_online(self, wiki_title):
        cur = self.conn.cursor()
        sql = "SELECT content FROM wiki_online WHERE entity_id=(E\'%s\')"
        cur.execute(sql % wiki_title.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        # print(row)
        return row

    def fetch_entity_context_from_tackbp(self, wiki_title):
        cur = self.conn.cursor()
        sql = "SELECT wiki_text FROM kb_tac2014 WHERE wiki_title=(E\'%s\')"
        cur.execute(sql % wiki_title.replace("'", "\\'"))
        row = cur.fetchone()
        # print(row)
        cur.close()
        return row

    def fetch_negative_training_cases(self, mention, entity):
        cur = self.conn.cursor()
        sql = "Select DISTINCT(entity) from wiki_training WHERE wiki_training.candidate_size_aida>9 AND wiki_training.surfaceform=(E\'%s\') AND wiki_training.entity <> (E\'%s\') AND wiki_training.id IN (SELECT doc_id FROM training_processed_clauses);"
        cur.execute(sql % (mention.replace("'", "\\'"), entity.replace("'", "\\'")))
        rows = cur.fetchall()
        return rows

    def fetch_all_entity_from_processed_clauses(self):
        cur = self.conn.cursor()
        sql = "Select DISTINCT(entity) from wiki_training WHERE wiki_training.candidate_size_aida>9 ;"
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def fetch_all_entity_from_conll_testb(self):
        cur = self.conn.cursor()
        sql = "Select DISTINCT(annotation) from aida_conll WHERE aida_conll.type=\'testb\' AND annotation!='NIL';"
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def fetch_all_entity_from_wiki_training(self):
        cur = self.conn.cursor()
        sql = "Select wiki_training.surfaceform, wiki_training.entity from wiki_training WHERE wiki_training.candidate_size_aida>9 AND wiki_training.id IN (SELECT doc_id FROM training_processed_clauses);"
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def get_all_words_idf_dict(self):
        word_tuples = self.fetch_all_words_idf()
        word_idf_dict = dict(word_tuples)
        # self.db.close_db_conn()
        return word_idf_dict

    def get_all_words_mi_dict(self):
        word_tuples = self.fetch_all_words_mi()
        word_mi_dict = {(a,b):c for a,b,c in word_tuples}
        # self.db.close_db_conn()
        return word_mi_dict

    def get_collection_size(self):
        collection_size = self.fetch_collection_size()
        return collection_size[0][0]

    def fetch_all_words_idf(self):
        cur = self.conn.cursor()
        sql = "SELECT word_count.word, idf FROM word_count"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_words_mi(self):
        cur = self.conn.cursor()
        sql = "SELECT entity, word, mi FROM word_mi"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_collection_size(self):
        cur = self.conn.cursor()
        sql = "select value from meta WHERE key='collection_size'"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_men_articles(self):
        cur = self.conn.cursor()
        sql = "select aida_conll.doc_id, aida_conll.doc_text from aida_conll;"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_men_articles_source(self, dataset_source):
        cur = self.conn.cursor()
        sql = "select DISTINCT doc_id, doc_text from %s;"
        cur.execute(sql % dataset_source)
        rows = cur.fetchall()
        cur.close()
        return rows

