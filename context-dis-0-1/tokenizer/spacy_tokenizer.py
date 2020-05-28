import math
from collections import defaultdict

import logging_config
from const_vars.const_vars_abc import Dictionaries

log = logging_config.get_logger()


def create_index(tokens):
    index = defaultdict(list)
    for token_index, token in enumerate(tokens):
        index[token].append(token_index)
    return index


def tokenize_keyphrase(keyphrase):
    doc = Dictionaries.spacy_init.tokenizer(keyphrase)
    Dictionaries.spacy_init.tagger(doc)
    return [token.lemma_ for token in doc]


def main_gen_tf_idf_entropy_fea_redis(ents, query_id, doc_id, mention):
    men_toks_idx = Dictionaries.redis_db_obj.get_men_tokens_redis(Dictionaries.dataset_source, doc_id, redis_key="men-tok-ner-spacy-")
    if not men_toks_idx:
        log.error("No ctx for mention!!")
        raise Exception
    ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_dict = \
        gen_ent_tokens_tf_idf_entropy_fea_redis(ents, mention)
    return ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_dict, men_toks_idx


def gen_ent_tokens_tf_idf_entropy_fea_redis(ents, mention):
    # check whether exist in the redis
    idf_dict = Dictionaries.redis_db_obj.get_men_ents_tokens_idf_redis(mention, redis_key='sm50')
    entropy_dict = Dictionaries.redis_db_obj.get_men_ents_tokens_entropy_redis(mention, redis_key='sm50')
    tf_dict = Dictionaries.redis_db_obj.get_men_ents_tokens_tf_redis(mention, redis_key='sm50')
    if idf_dict and entropy_dict and tf_dict:
        return entropy_dict, idf_dict, tf_dict
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
        ent_toks_dict = Dictionaries.redis_db_obj.\
            get_ent_tokens_redis(ent, redis_key=Dictionaries.ent_repo_id)
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
    tok_idf_dict = defaultdict(float)
    ent_tok_idf_dict = defaultdict(dict)
    ent_tok_entropy_dict = defaultdict(dict)
    ent_tok_tf_dict = defaultdict(dict)

    for tok_idx, tok in enumerate(m_toks):
        n = 0
        for _, tok_fq_count in tok_fq_count_dict[tok].items():
            if tok_fq_count:
                n += 1
                p_fq_tok = tok_fq_count / len(ents)
                tok_entropy_dict[tok] -= p_fq_tok * math.log10(p_fq_tok)
        # normalize entropy
        if n > 1:
            tok_entropy_dict[tok] /= math.log10(n)

        tok_idf_dict[tok] = math.log10(d_n / e_toks_count[tok])

        for ent_idx, ent in enumerate(ents):
            ent_tok_dict = ent_tok_idx_dict[ent]
            if tok in ent_tok_dict:
                ent_tok_idf_dict[ent][tok] = tok_idf_dict[tok]
                ent_tok_tf_dict[ent][tok] = len(ent_tok_dict[tok]) / cal_fq(ent_tok_dict)
                ent_tok_entropy_dict[ent][tok] = tok_entropy_dict[tok]

    Dictionaries.redis_db_obj.save_men_ents_tokens_entropy_redis(mention, ent_tok_entropy_dict, redis_key='sm50')
    Dictionaries.redis_db_obj.save_men_ents_tokens_idf_redis(mention, ent_tok_idf_dict, redis_key='sm50')
    Dictionaries.redis_db_obj.save_men_ents_tokens_tf_redis(mention, ent_tok_tf_dict, redis_key='sm50')
    return ent_tok_entropy_dict, ent_tok_idf_dict, ent_tok_tf_dict


def cal_fq(tok_idxs_dict):
    total = 0
    for tok, idxs in tok_idxs_dict.items():
        total += len(idxs)
    return total


if __name__ == '__main__':
    # Dictionaries.spacy_init = spacy.load('en')
    Dictionaries.init_dictionaries()
    # ents = ['Japan', 'Beijing']
    # ss = Dictionaries.spacy_init.tokenizer('Beijing is a city')
    # print(ss)
    # ent_article_tok(ents)
    # texts = 'Beijig is a great city, EB, Japanese, Asia, Japanese, Asia, Japanese, Asia, Japanese, Asia, Japanese, Asia, Pacific sdfdf     Japan japnasfd dfa Japan.'
    # ent_tok_dict, men_toks_idx = ent_article_tok(ents), men_toks('Beijing', texts)
    # gen_comm_tokens_web(ent_tok_dict, men_toks_idx)
    # ent_articles_tok_redis()
    # men_articles_tok_redis()
    # ent_articles_tok_redis_single_test()
