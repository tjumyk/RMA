import json
import psycopg2
import os
import task_scheduler
from task_scheduler import logging_config
from typing import List

log = logging_config.getLogger()


# a class for processing the user input document and store the results in psql
class PostgreSQLDB(object):
    def __init__(self):
        # get the config path
        path = os.path.dirname(task_scheduler.__path__[0])
        config_path = os.path.join(path, 'config', 'config.json')
        log.info('config.path: %s' % config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        db = config['db']

        self.conn = psycopg2.connect(database=db['database'], user=db['user'], password=db['password'], host=db['host'],
                                     port=db['port'])

    def fetch_tasks_done(self):
        cur = self.conn.cursor()
        sql = "select entity from task_scheduler;"
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def update_tasks_queue(self, wiki_title):
        cur = self.conn.cursor()
        sql = "update task_scheduler set done=TRUE WHERE entity=(E\'%s\');"
        cur.execute(sql % wiki_title.replace("'", "\\'"))
        self.conn.commit()
        cur.close()

    def save_new_task(self, wiki_title, priority):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO task_scheduler(entity, priority) VALUES (%s, %s)", (wiki_title, priority))
        self.conn.commit()
        cur.close()

    def fetch_processed_mentions(self):
        cur = self.conn.cursor()
        cur.execute("Select DISTINCT mention from tasks")
        row = cur.fetchall()
        cur.close()
        return row

    def save_new_tasks_batch(self, tasks):
        cur = self.conn.cursor()
        args_str = ','.join(cur.mogrify("(%s,%s, %s)", x).decode('utf-8') for x in tasks)
        print(args_str)
        # sql = "INSERT into word_count (word, count, idf) VALUES (" + ",".join(["%s"] * len(word_items)) + ")"
        cur.execute("INSERT into tasks(entity, mention, source) VALUES " + args_str)
        self.conn.commit()
        cur.close()

    def fetch_inlink_dict(self, entity_id):
        cur = self.conn.cursor()
        sql = "SELECT inlink_dict.inlink_entity FROM inlink_dict WHERE entity=(E\'%s\');"
        cur.execute(sql % entity_id.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        return row

    def fetch_wiki_article(self, wiki_id):
        cur = self.conn.cursor()
        sql = "SELECT content FROM wikipedia_articles WHERE entity_id=(E\'%s\')"
        cur.execute(sql % wiki_id.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        return row

    def save_wiki_article(self, wiki_info):
        cur = self.conn.cursor()
        sql = "INSERT INTO wikipedia_articles (entity_id, url, content, length) VALUES (" + ",".join(
            ["%s"] * len(wiki_info)) + ")"
        cur.execute(sql, wiki_info)
        self.conn.commit()
        cur.close()

    def save_coref_input(self, coreference=None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO input_coref (start_position, end_position, is_representative_mention, sentence_number, doc_id,"
            " word_id, coref_id ) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (coreference['characterOffsetBegin'], coreference['characterOffsetEnd'],
             coreference['isRepresentativeMention'],
             coreference['sentNum'] - 1, coreference['doc_id'], coreference['text'], coreference['coref_id']))
        self.conn.commit()
        cur.close()

    def fetch_wiki_sentences(self, wiki_title):
        cur = self.conn.cursor()
        sql = "SELECT DISTINCT (sentence)FROM wiki_clauseie"
        cur.execute(sql)
        row = cur.fetchone()
        cur.close()
        return row[0]

    def save_coref_wiki(self, coreference=None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO wiki_coref (start_position, end_position, is_representative_mention, sentence_number, doc_id,"
            " word_id, coref_id ) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (coreference['characterOffsetBegin'], coreference['characterOffsetEnd'],
             coreference['isRepresentativeMention'],
             coreference['sentNum'] - 1, coreference['doc_id'], coreference['text'], coreference['coref_id']))
        self.conn.commit()
        cur.close()

    def save_coref_training(self, coreference=None):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO training_coref (start_position, end_position, is_representative_mention, sentence_number, doc_id,"
            " word_id, coref_id ) VALUES (%s, %s, %s, %s, %s, %s, %s)",
            (coreference['characterOffsetBegin'], coreference['characterOffsetEnd'],
             coreference['isRepresentativeMention'],
             coreference['sentNum'] - 1, coreference['doc_id'], coreference['text'], coreference['coref_id']))
        self.conn.commit()
        cur.close()

    def fetch_clause_number(self):
        cur = self.conn.cursor()
        sql = "SELECT COUNT(*) FROM wiki_clauseie"
        cur.execute(sql)
        row = cur.fetchone()
        cur.close()
        return row[0]

    def save_clausie_input(self, clausie):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO wiki_clauseie (subject_str, subject_position, relation_str, relation_position, object_str,object_position, sentence_id, doc_id, clause_depth, child_clause_id) VALUES (%s, %s, %s, %s, %s, %s, %s, %s,%s,%s);",
            (clausie['subject_str'], clausie['subject_position'], clausie['relation_str'], clausie['relation_position'],
             clausie['object_str'], clausie['object_position'], clausie['sentence_id'], clausie['doc_id'],
             clausie['clause_depth'], clausie['child_clause_id']))
        self.conn.commit()
        cur.close()

    def fetch_related_entity(self, entity_id):
        cur = self.conn.cursor()
        sql = "SELECT * FROM wiki_clausie_entity WHERE doc_id=%s"
        cur.execute(sql % (entity_id))
        rows = cur.fetchall()
        return rows

    def fetch_clausie_by_mention(self, mentions):
        cur = self.conn.cursor()
        # sql = "SELECT * FROM wiki_clauseie WHERE doc_id=%s"
        sql = "SELECT * FROM input_processed_clauses WHERE subject_str IN (%s) OR input_processed_clauses.subject_coref IN (%s) OR input_processed_clauses.object_coref IN (%s) OR input_processed_clauses.object_str IN (%s)"
        cur.execute(sql % (mentions, mentions, mentions, mentions))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_wikipedia_article(self, entity_id):
        pass

    def fetch_coref_by_word_input(self, doc_id, word, sentence_id, word_position):
        # print(doc_id)
        # print("doc_id:" + doc_id + "  word : " + word + " sentence_id " + str(sentence_id) + ' wp ' + str(word_position))
        cur = self.conn.cursor()
        sql = "SELECT id, word_id FROM input_coref WHERE doc_id =(E\'%s\')  AND is_representative_mention=TRUE " \
              "AND coref_id IN (SELECT coref_id FROM input_coref WHERE sentence_number=\'%s\' AND doc_id=(E\'%s\')" \
              " AND start_position=%s LIMIT 1)"
        cur.execute(sql % (doc_id.replace("'", "\\'"), str(sentence_id), doc_id.replace("'", "\\'"), word_position))
        row = cur.fetchone()
        # print(row)
        cur.close()
        return (-1, word) if row is None else row

    def fetch_wiki_coref_by_word(self, doc_id, word, sentence_id, word_position):
        # print(doc_id)
        # print("doc_id:" + doc_id + "  word : " + word + " sentence_id " + str(sentence_id) + ' wp ' + str(word_position))
        cur = self.conn.cursor()
        sql = "SELECT id, word_id FROM wiki_coref WHERE doc_id =(E\'%s\')  AND is_representative_mention=TRUE " \
              "AND coref_id IN (SELECT coref_id FROM wiki_coref WHERE sentence_number=\'%s\' AND doc_id=(E\'%s\')" \
              " AND start_position=%s LIMIT 1)"
        cur.execute(sql % (doc_id.replace("'", "\\'"), str(sentence_id), doc_id.replace("'", "\\'"), word_position))
        row = cur.fetchone()
        # print(row)
        cur.close()
        return (-1, word) if row is None else row

    def save_processed_clause_input(self, result):
        # print(result)
        cur = self.conn.cursor()
        sql = "INSERT INTO input_processed_clauses (subject_str, subject_coref, relation_str, relation_lemma, object_str, object_coref, subject_id, object_id, doc_id, sentence_id) VALUES (" + ",".join(
            ["%s"] * len(result)) + ")"
        # print(sql)
        cur.execute(sql, result)
        self.conn.commit()
        cur.close()

    def save_keyphrase_wiki(self, result):
        print(result)
        cur = self.conn.cursor()
        sql = "INSERT INTO wiki_processed_clauses (subject_str, subject_coref, relation_str, relation_lemma, object_str, object_coref, subject_id, object_id, entity_id, sentence_id) VALUES (" + ",".join(
            ["%s"] * len(result)) + ")"
        # print(sql)
        cur.execute(sql, result)
        self.conn.commit()
        cur.close()

    def save_sentence_wiki(self, sent_info):
        cur = self.conn.cursor()
        sql = "INSERT INTO wiki_sentences (sentence, sentence_id, doc_id) VALUES (" + ",".join(
            ["%s"] * len(sent_info)) + ")"
        cur.execute(sql, sent_info)
        self.conn.commit()
        cur.close()

    def fetch_entity_keyphrase_number_with_coref(self, entity):
        cur = self.conn.cursor()
        log.info('Entity: %s' % entity)
        sql = "SELECT count(*) FROM wiki_processed_clauses WHERE entity_id=(E\'%s\');"
        cur.execute(sql % entity.replace("'", "\\'"))
        rows = cur.fetchone()
        cur.close()
        return rows

    def save_sentence_training(self, sent_info):
        cur = self.conn.cursor()
        sql = "INSERT INTO training_sentences (sentence, sentence_id, doc_id) VALUES (" + ",".join(
            ["%s"] * len(sent_info)) + ")"
        cur.execute(sql, sent_info)
        self.conn.commit()
        cur.close()

    def save_sentence_input(self, sent_info):
        cur = self.conn.cursor()
        sql = "INSERT INTO input_sentences (sentence, sentence_id, doc_id) VALUES (" + ",".join(
            ["%s"] * len(sent_info)) + ")"
        cur.execute(sql, sent_info)
        self.conn.commit()
        cur.close()

    def fetch_entity_keyphrases(self, entity):
        cur = self.conn.cursor()
        log.info('Entity: %s' % entity)
        sql = "SELECT * FROM wiki_processed_clauses WHERE entity_id = ANY(%s);"
        cur.execute(sql, ([entity],))
        rows = cur.fetchall()
        cur.close()
        # print(rows)
        return rows

    def fetch_multi_entities_keyphrases(self, entities):
        cur = self.conn.cursor()
        log.info('Entities: %s' % str(entities))
        sql = "SELECT * FROM wiki_processed_clauses WHERE entity_id IN %s;"
        cur.execute(sql, (entities,))
        # cur.mogrify(sql, (entities,))
        rows = cur.fetchall()
        cur.close()
        # print(rows)
        return rows

    def fetch_mention_keyphrase(self, mention):
        cur = self.conn.cursor()
        log.info('Mention: %s ' % mention)
        sql = "SELECT * FROM input_processed_clauses WHERE doc_id = ANY(%s);"
        # log.info('SQL for mention keyphrases: %s' % (sql, ([mention],)))
        cur.execute(sql, ([mention],))
        rows = cur.fetchall()
        cur.close()
        # print(rows)
        return rows

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
        sql = "SELECT value FROM meta WHERE key='collection_size'"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_wiki_coref_by_doc_id(self, doc_id):
        cur = self.conn.cursor()
        sql = "SELECT * FROM wiki_coref WHERE doc_id =(E\'%s\')"
        cur.execute(sql % doc_id.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        return row

    def fetch_training_cases(self):
        cur = self.conn.cursor()
        sql = "SELECT id, doc_text FROM wiki_training WHERE wiki_training.candidate_size_aida>9"
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def fetch_training_cases_no_coref(self):
        cur = self.conn.cursor()
        sql = "SELECT wiki_training.id, wiki_training.doc_text FROM wiki_training WHERE wiki_training.candidate_size_aida>9 AND wiki_training.id NOT IN (SELECT DISTINCT (doc_id) FROM training_coref)"
        cur.execute(sql)
        rows = cur.fetchall()
        return rows

    def fetch_training_mention_coref_by_doc_id(self, doc_id):
        cur = self.conn.cursor()
        sql = "SELECT * FROM training_coref WHERE doc_id =%s"
        cur.execute(sql % doc_id)
        row = cur.fetchone()
        cur.close()
        return row

    def fetch_all_mention_conll_non_nil_testb(self):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_ace_non_nil(self):
        cur = self.conn.cursor()
        #sql = 'SELECT surfaceform FROM ace2014_uiuc WHERE annotation!=\'none\''
        sql = 'SELECT surfaceform FROM ace2004_wned WHERE annotation!=\'none\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_msnbc_non_nil(self):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM msnbc_new WHERE annotation!=\'none\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_clueweb_non_nil(self):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM clueweb12 WHERE annotation!=\'none\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_rss_non_nil(self):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM rss WHERE annotation!=\'none\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_wiki_wned_non_nil(self):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM wiki_wned WHERE annotation!=\'none\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_aquaint_non_nil(self):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM aquaint_new WHERE annotation!=\'none\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_uiuc_wiki_non_nil(self):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM wiki_uiuc WHERE annotation!=\'none\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\''
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_all_mention_conll_non_nil_testb_by_ids(self, ids):
        cur = self.conn.cursor()
        sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'testb\' AND id IN %s;'
        # sql = 'SELECT surfaceform FROM aida_conll WHERE annotation!=\'NIL\' AND type=\'train\''
        cur.execute(sql, (ids,))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_mentions_kore(self) -> List[tuple]:
        '''

        :return: List of Tuples
        '''
        cur = self.conn.cursor()
        sql = "SELECT surfaceform FROM kore WHERE annotation != 'NIL'"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_ground_truth_entity_by_mention(self, dataset_source, mention):
        cur = self.conn.cursor()
        sql = "SELECT id, surfaceform, annotation FROM %s WHERE surfaceform=(E\'%s\') AND annotation != 'none' AND annotation !='NIL'"
        cur.execute(sql % (dataset_source, mention.replace("'", "\\'")))
        row = cur.fetchone()
        cur.close()
        return row

    def close_db_conn(self):
        self.conn.close()

