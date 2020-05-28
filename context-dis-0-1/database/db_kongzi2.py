import psycopg2
import json
import database
import os
import logging_config

log = logging_config.get_logger()


# config_path = database.__path__[0]
class PostgreSQLAIDA(object):
    def __init__(self):
        config_path = database.__path__[0]
        config_path = os.path.join(config_path, 'config.json')
        # log.info('config.path: %s' % config_path)
        with open(config_path, 'r') as f:
            config = json.load(f)
        db = config['aida_db']

        self.conn = psycopg2.connect(database=db['database'], user=db['user'], password=db['password'], host=db['host'],
                                     port=db['port'])

    def fetch_entity_by_mention(self, mention):
        # print(mention)
        cur = self.conn.cursor()
        # do a PostgreSQL join to select the entity namestring from the tables dictionary and entity_ids
        sql = "SELECT entity_ids_escaped.escaped_entity, dictionary.prior FROM dictionary LEFT JOIN entity_ids_escaped ON dictionary.entity=entity_ids_escaped.entity_id WHERE dictionary.mention = (E\'%s\') ORDER BY dictionary.prior DESC;"
        cur.execute(sql % mention.replace("'", "\\'"))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_entity_by_mention_emnlp17(self, mention):
        # print(mention)
        cur = self.conn.cursor()
        # do a PostgreSQL join to select the entity namestring from the tables dictionary and entity_ids
        sql = "SELECT entity, prior FROM men_ent_dict_emnlp2017_with_yago WHERE men_ent_dict_emnlp2017_with_yago.mention = (E\'%s\') ORDER BY prior DESC;"
        cur.execute(sql % mention.replace("'", "\\'"))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_candidate_size_aida(self, mention):
        cur = self.conn.cursor()
        sql = "SELECT count(*) from dictionary WHERE mention=(E\'%s\');"
        cur.execute(sql % mention.replace("'", "\\'"))
        rows = cur.fetchone()
        cur.close()
        return rows

    def fetch_entity_by_mention_size(self, mention, size=50):
        # print(mention)
        cur = self.conn.cursor()
        if str(size) == '0':
            return self.fetch_entity_by_mention(mention)
        else:
            # do a PostgreSQL join to select the entity namestring from the tables dictionary and entity_ids
            sql = "SELECT entity_ids_escaped.escaped_entity, dictionary.prior FROM dictionary LEFT JOIN entity_ids_escaped ON dictionary.entity=entity_ids_escaped.entity_id WHERE dictionary.mention = (E\'%s\') ORDER BY dictionary.prior DESC LIMIT %s;"
            cur.execute(sql % (mention.replace("'", "\\'"), size))
            rows = cur.fetchall()
            cur.close()
            return rows

    def fetch_entity_by_mention_size_emnlp17(self, mention, size=50):
        # print(mention)
        cur = self.conn.cursor()
        if str(size) == '0':
            return self.fetch_entity_by_mention_emnlp17(mention)
        else:
            # do a PostgreSQL join to select the entity namestring from the tables dictionary and entity_ids
            sql = "SELECT entity, prior FROM men_ent_dict_emnlp2017_with_yago WHERE men_ent_dict_emnlp2017_with_yago.mention = (E\'%s\') ORDER BY prior DESC LIMIT %s;"
            cur.execute(sql % (mention.replace("'", "\\'"), size))
            rows = cur.fetchall()
            cur.close()
            return rows

    def fetch_entity_priors_by_mentions(self, mentions):
        # print(mention)
        cur = self.conn.cursor()
        # do a PostgreSQL join to select the entity namestring from the tables dictionary and entity_ids
        sql = "SELECT entity_ids_escaped.escaped_entity, dictionary.prior FROM dictionary LEFT JOIN entity_ids_escaped ON dictionary.entity=entity_ids_escaped.entity_id WHERE dictionary.mention IN %s;"
        cur.execute(sql, (mentions,))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_entity_popularity_given_mention(self, mention, entity_id):
        cur = self.conn.cursor()
        sql = "SELECT dictionary.prior FROM dictionary  WHERE dictionary.mention = (E\'%s\') and dictionary.entity=%s;"
        cur.execute(sql % (mention.replace("'", "\\'"), entity_id))
        row = cur.fetchone()
        cur.close()
        return row

    def fetch_aida_form_by_entity(self, entity):
        cur = self.conn.cursor()
        sql = "SELECT entity_ids_escaped.entity_id from entity_ids_escaped WHERE escaped_entity=(E\'%s\')"
        cur.execute(sql % entity.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        return row

    def fetch_aida_form_(self):
        cur = self.conn.cursor()
        sql = "SELECT entity_ids_escaped.entity_id, entity_ids_escaped.escaped_entity from entity_ids_escaped;"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_aida_form(self):
        cur = self.conn.cursor()
        sql = "SELECT entity_ids_escaped.ori_entity, entity_ids_escaped.escaped_entity from entity_ids_escaped;"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_men_ent_prior(self):
        cur = self.conn.cursor()
        sql = "SELECT dictionary.mention, entity_ids_escaped.escaped_entity, dictionary.prior FROM dictionary LEFT JOIN entity_ids_escaped ON dictionary.entity=entity_ids_escaped.entity_id;"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_men_candidate_ent_by_mention(self):
        cur = self.conn.cursor()
        sql = "SELECT entity_ids_escaped.escaped_entity FROM dictionary LEFT JOIN entity_ids_escaped ON dictionary.entity=entity_ids_escaped.entity_id ORDER BY dictionary.prior LIMIT 50;"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_ents_size(self):
        cur = self.conn.cursor()
        sql = "SELECT SUM(count) FROM entity_counts;"
        cur.execute(sql)
        rows = cur.fetchone()
        cur.close()
        return rows

    def fetch_men_can_size(self, men):
        cur = self.conn.cursor()
        sql = "SELECT count(*) from dictionary WHERE mention=(E\'%s\');"
        cur.execute(sql % men.replace("'", "\\'"))
        row = cur.fetchone()
        cur.close()
        return row

    def fetch_ents_count(self):
        cur = self.conn.cursor()
        sql = "SELECT entity_ids_escaped.escaped_entity, entity_counts.count FROM entity_counts LEFT JOIN " \
              "entity_ids_escaped ON entity_counts.entity=entity_ids_escaped.entity_id;"
        cur.execute(sql)
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_ents_count_emnlp17(self, ent):
        cur = self.conn.cursor()
        sql = "SELECT sum(count) from men_ent_dict_emnlp2017 WHERE entity=%s;"
        cur.execute(sql, (ent,))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_max_men_prior(self, men):
        cur = self.conn.cursor()
        sql = "SELECT max(dictionary.prior) FROM dictionary WHERE mention=(E\'%s\');"
        cur.execute(sql % men.replace("'", "\\'"))
        rows = cur.fetchone()
        cur.close()
        return rows

    def fetch_inlinks_by_ent(self, ent):
        cur = self.conn.cursor()
        sql = "SELECT _id FROM wikipedia_links_2014 WHERE target=%s;"
        cur.execute(sql, (ent,))
        rows = cur.fetchall()
        cur.close()
        return rows

    # fetch an entity's outlinks with duplicates
    def fetch_outlinks_by_ent(self, ent):
        cur = self.conn.cursor()
        sql = "SELECT target FROM wikipedia_links_2014 WHERE _id=%s;"
        cur.execute(sql, (ent,))
        rows = cur.fetchall()
        cur.close()
        return rows

    def fetch_ent_redirects(self, ent):
        cur = self.conn.cursor()
        sql = "select target from wikipedia_redirects_2014 WHERE _id=%s;"
        cur.execute(sql, (ent,))
        rows = cur.fetchone()
        cur.close()
        return rows

    def fetch_ent_redirects_batch(self, ents):
        with self.conn.cursor() as cur:
            sql = "select _id, target from wikipedia_redirects_2014 WHERE _id in (%s)" % ','.join(['%s'] * len(ents))
            cur.execute(sql, tuple(ents))
            return cur.fetchall()

    def fetch_ent_counts(self, ent):
        cur = self.conn.cursor()
        sql = "select count from wikipedia_entity_counts_2014 WHERE _id=%s;"
        cur.execute(sql, (ent,))
        rows = cur.fetchone()
        cur.close()
        return rows

    def fetch_entity_by_mention_2014(self, mention):
        # print(mention)
        cur = self.conn.cursor()
        # do a PostgreSQL join to select the entity namestring from the tables dictionary and entity_ids
        sql = "SELECT entity, prior FROM men_ent_dict_wiki_2014 WHERE men_ent_dict_wiki_2014.mention = (E\'%s\');"
        cur.execute(sql % mention.replace("'", "\\'"))
        rows = cur.fetchall()
        cur.close()
        return rows
