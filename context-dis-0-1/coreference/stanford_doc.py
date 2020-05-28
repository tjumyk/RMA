#!/usr/bin/env python

import time
from const_vars.const_vars_abc import Dictionaries
import urllib.parse
import urllib.request
import requests


class Coreference(object):
    def __init__(self):
        self.db = Dictionaries.db_conn_obj

    def coref_input_doc(self, doc=None, doc_id=None, doc_id_hash=None, data_source=None):
        stanford_properties = {
            "annotators": "tokenize,ssplit,lemma,entitymentions,coref",
            "coref.algorithm": "statistical",
            "outputFormat": "json",
            'timeout': '1000000'
        }
        params = {
            "properties": stanford_properties,
            "pipelineLanguage": "en"
        }
        # print(≥urllib.parse.urlencode(params))
        stanford_url = 'http://kongzi3:9000/'
        req = requests.post(stanford_url, params=urllib.parse.urlencode(params), data=doc.encode('utf-8'))
        # print(req)
        response = req.json()
        sentences_entitymentions = [[entitymentions for entitymentions in sent_atom.get('entitymentions')] for
                                    sent_atom in response.get('sentences')]
        corefs = response.get('corefs')
        for coref_id, coref_values in corefs.items():
            for coref_value in coref_values:
                for entitymention in sentences_entitymentions[coref_value.get('sentNum') - 1]:
                    if entitymention.get('tokenBegin') == coref_value['startIndex'] - 1 and\
                                    entitymention.get('tokenEnd') == coref_value['endIndex'] - 1:
                        coref_value['characterOffsetBegin'] = entitymention.get('characterOffsetBegin')
                        coref_value['characterOffsetEnd'] = entitymention.get('characterOffsetEnd')
                        coref_value['doc_id'] = doc_id
                        coref_value['doc_id_hash'] = doc_id_hash
                        coref_value['data_source'] = data_source
                        coref_value['coref_id'] = coref_id
                        self.db.save_coref_doc(coref_value)


def men_articles_coref_psql(data_source):
    men_art_tuple = Dictionaries.db_conn_obj.fetch_all_men_articles_source(data_source)
    men_ent_dict = dict(men_art_tuple)
    doc_idx = 0
    coref = Coreference()
    for men, m_text in men_ent_dict.items():
        doc_idx += 1
        print("Processsing doc {}, progress {}/{} = {:.2f}".format(men, doc_idx, 500, doc_idx / 500))
        doc_id_hash = Dictionaries.doc_to_hash(m_text)
        coref.coref_input_doc(m_text, men, doc_id_hash, data_source)


if __name__ == '__main__':
    Dictionaries.init_dictionaries()
    men_articles_coref_psql('aida_conll')
