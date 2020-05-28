from const_vars.constant_conll_testb import Dict_Co
from collections import defaultdict
import spacy
import math
import csv
import os
import json
import logging_config
import urllib.parse
import urllib.request
import requests

log = logging_config.get_logger()


def tok_file(toks_dict, m_e_id, ty):
    if not os.path.exists(Dict_Co.res_path):
        os.makedirs(Dict_Co.res_path)
    with open(os.path.join(Dict_Co.res_path, str(m_e_id) + "--" + ty + '.csv'),
              'w') as f:
        writer = csv.writer(f)
        writer.writerow(['token', 'count'])
        for tok, idxs in toks_dict.items():
            writer.writerow([tok, len(idxs)])


def create_index(tokens):
    index = defaultdict(list)
    for token_index, token in enumerate(tokens):
        index[token].append(token_index)
    return index


def tokenize_keyphrase(keyphrase):
    doc = Dict_Co.spacy_init.tokenizer(keyphrase)
    Dict_Co.spacy_init.tagger(doc)
    return [token.lemma_ for token in doc]


def men_articles_tok__conll_redis():
    men_art_tuple = Dict_Co.db_conn_obj.fetch_all_men_articles()
    men_ent_dict = dict(men_art_tuple)
    articles = list(men_ent_dict.values())
    for men, m_text in men_ent_dict.items():
        men_tok = []
        log.info("Processsing entity {}".format(men))
        doc = Dict_Co.spacy_init(m_text)
        for tok in doc:
            if tok.ent_iob in (0, 2) and not (
                            tok.is_stop or tok.is_punct or tok.is_space or tok.is_bracket
            or tok.is_quote):
                men_tok.append(tok.lemma_)
            elif tok.ent_iob == 3:
                for ent in doc.ents:
                    if ent.start == tok.i:
                        men_tok.append(ent.lemma_)
        men_toks_idx = create_index(men_tok)
        ss = Dict_Co.redis_db_obj.save_dict_redis('men-tok-ner-spacy-conll',
                                                  men, json.dumps(men_toks_idx))
        print(ss)


def men_articles_ner_redis(data_source):
    men_art_tuple = Dict_Co.db_conn_obj.fetch_all_men_articles_source(
        data_source)
    men_ent_dict = dict(men_art_tuple)
    doc_idx = 0
    # men_ner_spacy_dict = defaultdict(dict)
    for men, m_text in men_ent_dict.items():
        ner_spacy_dict = defaultdict(list)
        doc_idx += 1
        # men_tok = []
        log.info(
            "Processsing doc {}, progress {}/{} = {:.2f}".format(men, doc_idx,
                                                                 230,
                                                                 doc_idx / 230))
        doc = Dict_Co.spacy_init(m_text)
        filter_ent_types = ('DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL')
        for ent in doc.ents:
            if ent.label_ not in filter_ent_types:
                ner_spacy_dict[ent.text].append(ent.start_char)
        ss = Dict_Co.redis_db_obj.save_dict_redis(
            'men-ner-spacy-' + data_source, men, json.dumps(ner_spacy_dict))
        print(ss)


def men_articles_ner_stanford_redis(data_source):
    men_art_tuple = Dict_Co.db_conn_obj.fetch_all_men_articles_source(
        data_source)
    men_ent_dict = dict(men_art_tuple)
    doc_idx = 0
    stanford_properties = {
        "annotators": "tokenize,ssplit,lemma,entitymentions",
        "outputFormat": "json",
        'timeout': '1000000'
    }
    params = {
        "properties": stanford_properties,
        "pipelineLanguage": "en"
    }
    stanford_url = 'http://localhost:9000/'

    # men_ner_spacy_dict = defaultdict(dict)
    for men, m_text in men_ent_dict.items():
        ner_spacy_dict = defaultdict(list)
        doc_idx += 1
        # men_tok = []
        log.info(
            "Processsing doc {}, progress {}/{} = {:.2f}".format(men, doc_idx,
                                                                 230,
                                                                 doc_idx / 230))

        req = requests.post(stanford_url, params=urllib.parse.urlencode(params),
                            data=m_text.encode('utf-8'))
        response = req.json()
        sentences_entitymentions = [
            [entitymentions for entitymentions in sent_atom.get('entitymentions')] for
            sent_atom in response.get('sentences')]

        filter_ent_types = ('DATE', 'TIME', 'PERCENT', 'MONEY', 'QUANTITY', 'ORDINAL', 'CARDINAL',
                            'NUMBER', 'DURATION')
        for entitymentions in sentences_entitymentions:
            for entitymention in entitymentions:
                if entitymention.get('ner') not in filter_ent_types:
                    ner_spacy_dict[entitymention.get('text')].\
                        append(entitymention.get('characterOffsetBegin'))
        ss = Dict_Co.redis_db_obj.save_dict_redis(
            'men-ner-stanford-' + data_source, men, json.dumps(ner_spacy_dict))
        print(ss)


def ner_input_doc(self, doc=None, doc_id=None, doc_id_hash=None, data_source=None):

    req = requests.post(stanford_url, params=urllib.parse.urlencode(params), data=doc.encode('utf-8'))
    response = req.json()
    sentences_entitymentions = [[entitymentions for entitymentions in sent_atom.get('entitymentions')] for
                                sent_atom in response.get('sentences')]
    for entitymentions in sentences_entitymentions:
        for entitymention in entitymentions:
            if entitymention.get('tokenBegin') == coref_value['startIndex'] - 1 and \
                            entitymention.get('tokenEnd') == coref_value['endIndex'] - 1:
                self.db.save_coref_doc(coref_value)


def men_articles_tok_redis(data_source):
    men_art_tuple = Dict_Co.db_conn_obj.fetch_all_men_articles_source(
        data_source)
    men_ent_dict = dict(men_art_tuple)
    doc_idx = 0
    for men, m_text in men_ent_dict.items():
        doc_idx += 1
        men_tok = []
        log.info(
            "Processsing doc {}, progress {}/{} = {:.2f}".format(men, doc_idx,
                                                                 140,
                                                                 doc_idx / 140))
        doc = Dict_Co.spacy_init(m_text)
        for tok in doc:
            if tok.ent_iob in (0, 2) and not (
                            tok.is_stop or tok.is_punct or tok.is_space or tok.is_bracket or
                            tok.is_quote):
                men_tok.append(tok.lemma_)
            elif tok.ent_iob == 3:
                for ent in doc.ents:
                    if ent.start == tok.i:
                        men_tok.append(ent.lemma_)
        men_toks_idx = create_index(men_tok)
        ss = Dict_Co.redis_db_obj.save_dict_redis(
            'men-tok-ner-spacy-' + data_source, men, json.dumps(men_toks_idx))
        # print(ss)


def ent_articles_tok_redis():
    log.info("Loading articles from redis...")
    ent_art_dict = Dict_Co.redis_db_obj.conn.hgetall('entity_article_training')
    log.info("Finished Loading articles from redis.")
    articles = []
    doc_ent_dict = defaultdict(str)
    for ent, art in ent_art_dict.items():
        doc_ent_dict[art] = ent
        articles.append(art)
    ent_tok_dict = defaultdict(list)
    ent_tok_idx_dict = defaultdict(str)
    for doc in Dict_Co.spacy_init.pipe(articles, batch_size=10000,
                                       n_threads=Dict_Co.threads):
        cur_ent = doc_ent_dict[doc.text]
        log.info("Processsing entity {}".format(cur_ent))
        for tok in doc:
            if tok.ent_iob in (0, 2) and not (
                            tok.is_stop or tok.is_punct or tok.is_space or tok.is_bracket
            or tok.is_quote):
                ent_tok_dict[cur_ent].append(tok.lemma_)
            elif tok.ent_iob == 3:
                for ent in doc.ents:
                    if ent.start == tok.i:
                        ent_tok_dict[cur_ent].append(ent.lemma_)
        ent_tok_idx = create_index(ent_tok_dict[cur_ent])
        ent_tok_idx_dict[cur_ent] = json.dumps(ent_tok_idx)
    ss = Dict_Co.redis_db_obj.conn.hmset('ent-tok-ner-spacy', ent_tok_idx_dict)
    print(ss)


def ent_articles_tok_redis_single():
    log.info("Loading articles from redis...")
    ent_art_dict = Dict_Co.redis_db_obj.conn.hgetall('entity_article_training')
    log.info("Finished Loading articles from redis.")
    for ent, art in ent_art_dict.items():
        doc = Dict_Co.spacy_init(art)
        ent_tok_list = []
        log.info("Processing entity {}".format(ent))
        for tok in doc:
            if tok.ent_iob in (0, 2) and not (
                            tok.is_stop or tok.is_punct or tok.is_space or tok.is_bracket
            or tok.is_quote):
                ent_tok_list.append(tok.lemma_)
            elif tok.ent_iob == 3:
                for ent_ in doc.ents:
                    if ent_.start == tok.i:
                        ent_tok_list.append(ent_.lemma_)
        ent_tok_idx = create_index(ent_tok_list)
        log.info("Saving entity {}".format(ent))
        ss = Dict_Co.redis_db_obj.save_dict_redis('ent-tok-ner-spacy', ent,
                                                  json.dumps(ent_tok_idx))
        print(ss)


def ent_articles_tok_redis_single_test():
    log.info("Loading articles from redis...")
    ent_art_dict = \
    Dict_Co.redis_db_obj.conn.hmget('entity_article_training', 'Japan')[0]

    log.info("Finished Loading articles from redis.")
    # for ent, art in ent_art_dict.items():
    art = ent_art_dict
    ent = 'Japan'
    doc = Dict_Co.spacy_init(art)
    ent_tok_list = []
    log.info("Processsing entity {}".format(ent))
    for tok in doc:
        if tok.ent_iob in (0, 2) and not (
                        tok.is_stop or tok.is_punct or tok.is_space or tok.is_bracket
        or tok.is_quote):
            ent_tok_list.append(tok.lemma_)
        elif tok.ent_iob == 3:
            for ent_ in doc.ents:
                if ent_.start == tok.i:
                    ent_tok_list.append(ent_.lemma_)
    ent_tok_idx = create_index(ent_tok_list)
    ss = Dict_Co.redis_db_obj.save_dict_redis('ent-tok-ner-spacy', ent,
                                              json.dumps(ent_tok_idx))
    print(ss)


def ent_article_tok(ents):
    articles = []
    doc_ent_dict = defaultdict(str)
    for ent in ents:
        article = Dict_Co.redis_db_obj.get_ent_article_redis(
            'entity_article_training', ent)
        doc_ent_dict[article] = ent
        articles.append(article)

    ent_tok_dict = defaultdict(list)
    ent_tok_idx_dict = defaultdict(dict)
    for doc in Dict_Co.spacy_init.pipe(articles, batch_size=10000,
                                       n_threads=Dict_Co.threads):
        cur_ent = doc_ent_dict[doc.text]
        ent_tok_dict[cur_ent] = []
        start = 0
        for ent_span in doc.ents:
            ent_tok_dict[cur_ent] += [token.lemma_ for token in
                                      doc[start:ent_span.start]
                                      if not (
                token.is_stop or token.is_punct or token.is_space or token.is_bracket
                or token.is_quote)] + [ent_span.lemma_]
            # ents_spacy.append(ent_span.text)
            start = ent_span.end
        ent_tok_dict[cur_ent] += [token.lemma_ for token in doc[start:]]
        ent_tok_idx = create_index(ent_tok_dict[cur_ent])
        tok_file(ent_tok_idx, cur_ent, 'ent')
        ent_tok_idx_dict[cur_ent] = ent_tok_idx
    return ent_tok_idx_dict


def men_toks(men, m_text, w_size=False):
    men_toks_list = []
    start = 0
    doc = Dict_Co.spacy_init(m_text)
    for ent_span in doc.ents:
        men_toks_list += [token.lemma_ for token in doc[start:ent_span.start]
                          if not (
            token.is_stop or token.is_punct or token.is_space or token.is_bracket
            or token.is_quote)] + [ent_span.lemma_]
        # ents_spacy.append(ent_span.text)
        start = ent_span.end
    men_toks_list += [token.lemma_ for token in doc[start:]]
    men_lemm_list = [m_t.lemma_ for m_t in Dict_Co.spacy_init(men)]
    men_lemm = ' '.join(men_lemm_list)
    men_toks_idx = create_index(men_toks_list)
    if men_lemm not in men_toks_list or not w_size:
        tok_file(men_toks_idx, men, 'mention')
        return men_toks_idx
    else:
        men_toks_list_w_size = []
        m_idxs = men_toks_idx[men_lemm]
        for m_idx in m_idxs:
            start_idx = max(0, m_idx - w_size)
            end_idx = min(m_idx + w_size + 1, len(men_toks_list) - 1)
            men_toks_list_w_size += men_toks_list[start_idx: end_idx]
        men_toks_idx_w = create_index(men_toks_list_w_size)
        tok_file(men_toks_idx_w, men, 'mention')
        return men_toks_idx_w


def gen_comm_tokens(ent_tok_dict, men_toks_idx):
    m_toks = list(men_toks_idx.keys())
    ents = list(ent_tok_dict.keys())
    e_toks_count = []
    for ent, toks in ent_tok_dict.items():
        tok_array = []
        for tok in m_toks:
            if tok in toks:
                tok_array += [len(men_toks_idx[tok])]
        e_toks_count.append(tok_array)


def gen_comm_tokens_csv(ent_tok_dict, men_toks_idx):
    m_toks = list(men_toks_idx.keys())
    ents = list(ent_tok_dict.keys())
    with open(os.path.join(Dict_Co.res_path, 'co-occuer-matrix.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerow([' '] + ents)
        for tok in m_toks:
            tok_array = [tok]
            for ent in ents:
                if tok in ent_tok_dict[ent]:
                    tok_array += [
                        len(men_toks_idx[tok]) * len(ent_tok_dict[ent][tok])]
                else:
                    tok_array += [0]
            writer.writerow(tok_array)


def main_gen(ents, men, m_text):
    ent_tok_idx_dict, men_toks_idx = ent_article_tok(ents), men_toks(men,
                                                                     m_text)
    gen_comm_tokens_csv(ent_tok_idx_dict, men_toks_idx)


def main_gen_web(ents, men, m_text):
    ent_tok_idx_dict, men_toks_idx = ent_article_tok(ents), men_toks(men,
                                                                     m_text)
    result = gen_comm_tokens_web(ent_tok_idx_dict, men_toks_idx)
    return result


def main_gen_web_redis(ents, men, m_text):
    ent_tok_idx_dict, men_toks_idx = ent_article_tok(ents), men_toks(men,
                                                                     m_text)
    result = gen_comm_tokens_web(ent_tok_idx_dict, men_toks_idx)
    return result


def main_gen_web_co_redis(ents, men, m_text, doc_id):
    # ent_tok_idx_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis()
    men_toks_idx = Dict_Co.redis_db_obj.get_men_tokens_redis(doc_id)
    result = gen_comm_tokens_web_redis(ents, men_toks_idx)
    return result


def main_gen_web_fq_redis(ents, men, m_text, doc_id):
    # ent_tok_idx_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis()
    # men_toks_idx = Dict_Co.redis_db_obj.get_men_tokens_redis(doc_id)
    result = gen_ent_tokens_fq_web_redis(ents)
    return result


def main_gen_web_idf_entropy_redis(ents, men, m_text, doc_id):
    # ent_tok_idx_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis()
    # men_toks_idx = Dict_Co.redis_db_obj.get_men_tokens_redis(doc_id)
    result = gen_ent_tokens_idf_entropy_web_redis(ents)
    return result


def main_gen_tf_idf_entropy_fea_redis(ents, men, m_text, doc_id):
    men_toks_idx = Dict_Co.redis_db_obj.get_men_tokens_redis(doc_id)
    ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_idf_dict, ent_tok_tf_entropy_dict = \
        gen_ent_tokens_tf_idf_entropy_fea_redis(ents)
    return ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_idf_dict, ent_tok_tf_entropy_dict, men_toks_idx


def main_gen_web_idf_redis(ents, men, m_text, doc_id):
    result = gen_ent_tokens_idf_web_redis(ents)
    return result


def gen_ent_tokens_tf_idf_entropy_fea_redis(ents):
    # total tokens in all the documents
    m_toks = []
    # key: token; value: the number of documents containing a token
    e_toks_count = defaultdict(int)
    # tok_fq_dict = defaultdict(int)
    # key: token; value: {dict: key: fq; value: number of documents containing this frequency}
    tok_fq_count_dict = defaultdict(dict)
    # tok_tal = 0
    d_n = len(ents)
    ent_tok_idx_dict = defaultdict(dict)
    for ent in ents:
        # key: token, value: a list of indexs where the token occurs in the document
        ent_toks_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis(ent)
        for ent_tok_, idxs in ent_toks_dict.items():
            if len(idxs) in tok_fq_count_dict[ent_tok_]:
                tok_fq_count_dict[ent_tok_][len(idxs)] += 1
            else:
                tok_fq_count_dict[ent_tok_][len(idxs)] = 1
            e_toks_count[ent_tok_] += 1
        ent_tok_idx_dict[ent] = ent_toks_dict
        ent_tokens = ent_toks_dict.keys()
        m_toks += ent_tokens
    m_toks = list(set(m_toks))
    # calculate for frequency 0
    for ent_tok_ in m_toks:
        tok_fq_count_dict[ent_tok_][0] = len(ents) - e_toks_count[ent_tok_]

    tok_entropy_dict = defaultdict(float)
    ent_tok_tf_idf_dict = defaultdict(dict)
    tok_idf_dict = defaultdict(float)
    ent_tok_idf_dict = defaultdict(dict)
    ent_tok_entropy_dict = defaultdict(dict)
    ent_tok_tf_entropy_dict = defaultdict(dict)

    for tok_idx, tok in enumerate(m_toks):
        for _, tok_fq_count in tok_fq_count_dict[tok].items():
            if tok_fq_count:
                p_fq_tok = tok_fq_count / len(ents)
                tok_entropy_dict[tok] -= p_fq_tok * math.log10(p_fq_tok)

        tok_idf_dict[tok] = math.log10(d_n / e_toks_count[tok])

        for ent_idx, ent in enumerate(ents):
            ent_tok_dict = ent_tok_idx_dict[ent]
            if tok in ent_tok_dict:
                tf = len(ent_tok_dict[tok]) / cal_fq(ent_tok_dict)
                tfidf = tf * tok_idf_dict[tok]
                tfentropy = tf * tok_entropy_dict[tok]
                ent_tok_tf_entropy_dict[ent][tok] = tfentropy
                ent_tok_tf_idf_dict[ent][tok] = tfidf
                ent_tok_idf_dict[ent][tok] = tok_idf_dict[tok]
                ent_tok_entropy_dict[ent][tok] = tok_entropy_dict[tok]

    return ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_idf_dict, ent_tok_tf_entropy_dict


def gen_ent_tokens_idf_entropy_web_redis(ents):
    # total tokens in all the documents
    m_toks = []
    # key: token; value: the number of documents containing a token
    e_toks_idf = defaultdict(int)
    # tok_fq_dict = defaultdict(int)
    # key: token; value: {dict: key: fq; value: number of documents containing this frequency}
    tok_fq_count_dict = defaultdict(dict)
    # tok_tal = 0
    d_n = len(ents)
    # ent_tok_idx_dict = defaultdict(dict)
    for ent in ents:
        # key: token, value: a list of indexs where the token occurs in the document
        ent_toks_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis(ent)
        for ent_tok_, idxs in ent_toks_dict.items():
            if len(idxs) in tok_fq_count_dict[ent_tok_]:
                tok_fq_count_dict[ent_tok_][len(idxs)] += 1
            else:
                tok_fq_count_dict[ent_tok_][len(idxs)] = 1
            e_toks_idf[ent_tok_] += 1

            # tok_tal += len(idxs)
            # tok_fq_dict[ent_tok_] += len(idxs)
        # ent_tok_idx_dict[ent] = ent_toks_dict
        ent_tokens = ent_toks_dict.keys()
        m_toks += ent_tokens
    m_toks = list(set(m_toks))
    # calculate for frequency 0
    for ent_tok_ in m_toks:
        tok_fq_count_dict[ent_tok_][0] = len(ents) - e_toks_idf[ent_tok_]

    col_labels = ['IDF', 'Entropy']
    result = {'row_labels': m_toks, 'col_labels': col_labels,
              'hccol': [i + 1 for i in range(len(col_labels))],
              'hcrow': [i + 1 for i in range(len(m_toks))],
              'hccol_num': len(col_labels), 'hcrow_num': len(m_toks),
              'max_idf': 0, 'min_idf': 0}
    data = []

    for tok_idx, tok in enumerate(m_toks):
        tok_dict_idf = {'row': tok_idx + 1, 'col': 1,
                        'value': math.log10(d_n / e_toks_idf[tok])}
        tok_dict_etr = {'row': tok_idx + 1, 'col': 2, 'value': 0}
        for _, tok_fq_count in tok_fq_count_dict[tok].items():
            if tok_fq_count:
                p_fq_tok = tok_fq_count / len(ents)
                tok_dict_etr['value'] -= p_fq_tok * math.log10(p_fq_tok)
        data.append(tok_dict_idf)
        data.append(tok_dict_etr)
    result['data'] = data
    return result


def cal_fq(tok_idxs_dict):
    total = 0
    for tok, idxs in tok_idxs_dict.items():
        total += len(idxs)
    return total


def gen_ent_tokens_idf_web_redis(ents):
    # total tokens in all the documents
    m_toks = []
    e_toks_idf = defaultdict(int)
    d_n = len(ents)
    ent_tok_idx_dict = defaultdict(dict)
    for ent in ents:
        ent_toks_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis(ent)
        ent_tok_idx_dict[ent] = ent_toks_dict
        ent_tokens = ent_toks_dict.keys()
        m_toks += ent_tokens
        for m_tok_ in ent_tokens:
            e_toks_idf[m_tok_] += 1
    m_toks = list(set(m_toks))

    result = {'row_labels': m_toks, 'col_labels': ents,
              'hccol': [i + 1 for i in range(len(ents))],
              'hcrow': [i + 1 for i in range(len(m_toks))],
              'hccol_num': len(ents), 'hcrow_num': len(m_toks),
              'max_idf': 0, 'min_idf': 0}
    data = []

    for tok_idx, tok in enumerate(m_toks):
        for ent_idx, ent in enumerate(ents):
            tok_dict = {'row': tok_idx + 1, 'col': ent_idx + 1}
            ent_tok_dict = ent_tok_idx_dict[ent]
            if tok in ent_tok_dict:
                tfidf = len(ent_tok_dict[tok]) / cal_fq(
                    ent_tok_dict) * math.log2(d_n / e_toks_idf[tok])
                result['max_idf'] = max(result['max_idf'], tfidf)
                tok_dict['value'] = tfidf
            else:
                tok_dict['value'] = 0
            data.append(tok_dict)
    result['data'] = data
    return result


def gen_ent_tokens_fq_web_redis(ents):
    m_toks = []
    ent_tok_idx_dict = defaultdict(dict)
    for ent in ents:
        ent_toks_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis(ent)
        ent_tok_idx_dict[ent] = ent_toks_dict
        m_toks.extend(ent_toks_dict.keys())
    m_toks = list(set(m_toks))
    result = {'row_labels': m_toks, 'col_labels': ents,
              'hccol': [i + 1 for i in range(len(ents))],
              'hcrow': [i + 1 for i in range(len(m_toks))],
              'hccol_num': len(ents), 'hcrow_num': len(m_toks)}
    data = []

    for tok_idx, tok in enumerate(m_toks):
        for ent_idx, ent in enumerate(ents):
            tok_dict = {'row': tok_idx + 1, 'col': ent_idx + 1}
            ent_tok_dict = ent_tok_idx_dict[ent]
            if tok in ent_tok_dict:
                tok_dict['value'] = len(ent_tok_dict[tok])
            else:
                tok_dict['value'] = 0
            data.append(tok_dict)
    result['data'] = data
    return result


def gen_comm_tokens_web_redis(ents, men_toks_idx):
    m_toks = list(men_toks_idx.keys())
    result = {'row_labels': m_toks, 'col_labels': ents,
              'hccol': [i + 1 for i in range(len(ents))],
              'hcrow': [i + 1 for i in range(len(m_toks))],
              'hccol_num': len(ents), 'hcrow_num': len(m_toks),
              'max_co': 0, 'min_co': 0}
    data = []

    for tok_idx, tok in enumerate(m_toks):
        for ent_idx, ent in enumerate(ents):
            tok_dict = {'row': tok_idx + 1, 'col': ent_idx + 1}
            ent_tok_dict = Dict_Co.redis_db_obj.get_ent_tokens_redis(ent)
            if tok in ent_tok_dict:
                co_times = len(men_toks_idx[tok]) * len(ent_tok_dict[tok])
                tok_dict['value'] = co_times
                result['max_co'] = max(result['max_co'], co_times)
            else:
                tok_dict['value'] = 0
            data.append(tok_dict)
    result['data'] = data
    return result


def gen_comm_tokens_web(ent_tok_dict, men_toks_idx):
    m_toks = list(men_toks_idx.keys())
    ents = list(ent_tok_dict.keys())
    result = {'row_labels': m_toks, 'col_labels': ents}
    result['hccol'] = [i + 1 for i in range(len(ents))]
    result['hcrow'] = [i + 1 for i in range(len(m_toks))]
    result['hccol_num'] = len(ents)
    result['hcrow_num'] = len(m_toks)
    data = []

    for tok_idx, tok in enumerate(m_toks):
        for ent_idx, ent in enumerate(ents):
            tok_dict = {'row': tok_idx + 1, 'col': ent_idx + 1}
            if tok in ent_tok_dict[ent]:
                tok_dict['value'] = len(men_toks_idx[tok]) * len(
                    ent_tok_dict[ent][tok])
            else:
                tok_dict['value'] = 0
            data.append(tok_dict)
            # writer.writerow(tok_array)
    result['data'] = data
    # print(result)
    return result


if __name__ == '__main__':
    # Dict_Co.spacy_init = spacy.load('en')
    Dict_Co.init_dictionaries()
    # ents = ['Japan', 'Beijing']
    # ss = Dict_Co.spacy_init.tokenizer('Beijing is a city')
    # print(ss)
    # ent_article_tok(ents)
    # texts = 'Beijig is a great city, EB, Japanese, Asia, Japanese, Asia, Japanese, Asia, Japanese, Asia, Japanese, Asia, Pacific sdfdf     Japan japnasfd dfa Japan.'
    # ent_tok_dict, men_toks_idx = ent_article_tok(ents), men_toks('Beijing', texts)
    # gen_comm_tokens_web(ent_tok_dict, men_toks_idx)
    # ent_articles_tok_redis()
    # men_articles_tok_redis('kore')

    import sys
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    else:
        raise ValueError('Dataset name is required')
    print('Using dataset %s' % dataset_name)

    if len(sys.argv) > 2:
        action = sys.argv[2]
    else:
        raise ValueError('Action is required')
    print('Action: %s' % action)

    #men_articles_tok_redis(dataset_name)
    # men_articles_tok_redis('aquaint_new')
    # men_articles_ner_redis('aida_conll')
    if action == 'tok':
        men_articles_tok_redis(dataset_name)
    elif action == 'ner':
        men_articles_ner_stanford_redis(dataset_name)
    else:
        raise ValueError('invalid action: %s' % action)
    # men_articles_ner_stanford_redis('aquaint_new')
    # men_articles_ner_stanford_redis('ace2014_uiuc')

    # men_articles_tok_redis('msnbc_new')
    #men_articles_tok_redis('ace2014_uiuc')
    # ent_articles_tok_redis_single_test()
    # ent_articles_tok_redis_single()
