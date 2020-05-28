import json
from random import shuffle
import re
from collections import defaultdict
import spacy
import redis
import requests
import task_scheduler.constant
from task_scheduler.constant import Dictionaries
from bs4 import BeautifulSoup


nlp = spacy.load('en_core_web_sm')


def get_wiki_article(entity):
    result = ''
    header = {'User-Agent': 'Mozilla/5.0'}  # Needed to prevent 403 error on Wikipedia
    wiki_url = "https://en.wikipedia.org/wiki/"
    respond = requests.get(wiki_url + entity, headers=header)
    soup = BeautifulSoup(respond.text)
    for paragraph in soup.find_all("p"):
        res_text = re.sub(r'\[([0-9]+)\]', '', paragraph.text)
        result += res_text + '\n'
    return result


def get_wiki_article_local(entity):
    wiki_id = 'en.wikipedia.org/wiki/' + entity
    # wiki_id_re = Dictionaries.prior_db_conn_obj.fetch_ent_redirects(wiki_id)
    # wiki_id_re = wiki_id if not wiki_id_re else wiki_id_re[0]
    result = Dictionaries.prior_db_conn_obj.fetch_wiki_article(wiki_id)
    result = result[0] if result else ''
    # header = {'User-Agent': 'Mozilla/5.0'}  # Needed to prevent 403 error on Wikipedia
    # wiki_url = "https://en.wikipedia.org/wiki/"
    # respond = requests.get(wiki_url + entity, headers=header)
    # soup = BeautifulSoup(respond.text)
    # for paragraph in soup.find_all("p"):
    #     res_text = re.sub(r'\[([0-9]+)\]', '', paragraph.text)
    #     result += res_text + '\n'
    return result


def proess_new_wiki(ent):
    if conn.hexists('entity_article_training', ent):
        if conn.hexists('ent-tok-ner-spacy', ent):
            print("Already processed entity.")
        else:
            print("Tokenizing entity.")
            ent_article_tok_redis(ent)
            print("Done.")
    else:
        print("Gettting article.")
        wiki_article = get_wiki_article(ent)
        print("Saving entity.")
        conn.hset('entity_article_training', ent, wiki_article)
        print("Tokenizing entity.")
        ent_article_tok_redis(ent)
        print("Done.")


def proess_new_wiki_local(ent):
    wiki_id = 'en.wikipedia.org/wiki/' + ent
    wiki_id_re = Dictionaries.prior_db_conn_obj.fetch_ent_redirects(wiki_id)
    wiki_id_re = ent if not wiki_id_re else wiki_id_re[0].replace('en.wikipedia.org/wiki/', '')
    if not conn.hexists('ent-tok-ner-spacy-30-new', wiki_id_re):
        if ent != 'Arrow_(symbol)' and ent != 'Autostrada_A1_(Italy)':
            wiki_article = get_wiki_article_local(wiki_id_re)
            if wiki_article == "":
                print("Emppty article {}".format(ent.replace("\u2013", '')))
            print("processing new entity.")
            max_doc_length = 1000000
            if len(wiki_article) > max_doc_length:
                print('Document of "%s" is too big: %d characters. Truncated to %d.' % (wiki_id_re, len(wiki_article),
                                                                                        max_doc_length))
                wiki_article = wiki_article[:max_doc_length]
            ent_article_tok_redis(wiki_id_re, wiki_article, ent)
    # else:
    #     print("Already Done...")


with open('config/redis_config.json') as f_json:
    redis_config = json.load(f_json)['redis_db']
conn = redis.Redis(decode_responses=True, **redis_config)


def save_dict_redis(conn, redis_key, *dict_val):
    if len(dict_val) == 2:
        print('aaaaaaaaaaaa')
        return conn.hset(redis_key, dict_val[0], dict_val[1])
    if len(dict_val) == 1 and isinstance(dict_val[0], dict):
        print('ssssssss')
        return conn.hmset(redis_key, dict_val[0])
    else:
        raise Exception


def create_index(tokens):
    index = defaultdict(list)
    for token_index, token in enumerate(tokens):
        index[token].append(token_index)
    return index


def ent_article_tok_redis(ent, ent_art, ent_orig):
    tokens = []
    doc = nlp(ent_art)
    for tok in doc:
        if tok.ent_iob in (0, 2) and not (tok.is_stop or tok.is_punct or tok.is_space or tok.is_bracket
                                          or tok.is_quote):
            tokens.append(tok.lemma_)
        elif tok.ent_iob == 3:
            for ent_ in doc.ents:
                if ent_.start == tok.i:
                    tokens.append(ent_.lemma_)
    ent_tok_idx = create_index(tokens)

    dump = json.dumps(ent_tok_idx)
    conn.hset('ent-tok-ner-spacy-30-new', ent, dump)
    # conn.hset('ent-tok-ner-spacy-30-new', ent_orig, dump)
    # print(ss)


if __name__ == '__main__':
    task_scheduler.constant.init_dictionaries()
    ent_path = './candidate_ents_50.csv'
    # ent_path = './candidate_ents_conll_emnlp17_new_50.csv'
    # ent_path = './candidate_ents_wiki_uiuc.csv'
    ents = []
    with open(ent_path, 'r', encoding='utf-8') as f:
        ents = [x.strip('\n') for x in f.readlines()]

    ents = list(set(ents))
    shuffle(ents)
    for ent_idx, ent in enumerate(ents):
        if not ent_idx % 500:
            print("Processing about {:.2f}%".format(ent_idx / len(ents) * 100))
        proess_new_wiki_local(ent)

    # ent = 'Leeds_United_A.F.C.'
    # proess_new_wiki_local(ent)
