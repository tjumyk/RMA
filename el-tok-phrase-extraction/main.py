from const_vars.constant_conll_testb import Dict_Co
import tokenizer.spacy_tokenizer


Dict_Co.init_dictionaries()

if __name__ == '__main__':
    tokenizer.spacy_tokenizer.ent_articles_tok_redis_single()