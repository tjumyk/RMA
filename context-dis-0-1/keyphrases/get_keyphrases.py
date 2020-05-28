from const_vars.constant_conll_testb import Dict_Co as Dictionaries
from collections import defaultdict


def get_men_keyphrase_with_distance_dict(mention, doc_id, distance=1):
    men_keyp_with_dis = defaultdict(list)
    men_keyp_with_dis_no_dup = defaultdict(set)
    men_keyphrases_all = Dictionaries.men_keyphrase_dict[doc_id]

    for ctx_tuple in men_keyphrases_all:
        ctx_0 = []
        ctx_1 = []
        for item in ctx_tuple:
            if item.lower() != 'null':
                if item.lower() not in Dictionaries.stopwords and \
                        (mention.lower() in item.lower() or item.lower() in mention.lower()):
                    ctx_0.append(item)
                else:
                    ctx_1.append(item)
        # fixed the bug
        if ctx_0:
            men_keyp_with_dis[0] += ctx_0
            men_keyp_with_dis[1] += ctx_1
    men_keyp_with_dis_no_dup[0] = set(men_keyp_with_dis[0])
    men_keyp_with_dis_no_dup[1] = set(men_keyp_with_dis[1])
    return men_keyp_with_dis_no_dup


def get_ent_keyphrase_with_distance_dict(entity, distance=2):
    ent_keyp_with_dis = defaultdict(list)
    ent_keyp_with_dis_no_dup = defaultdict(set)
    ent_keyphrases_all = Dictionaries.get_entity_keyphrase(entity)
    for ctx_tuple in ent_keyphrases_all:
        ctx_0 = []
        ctx_1 = []
        for item in ctx_tuple:
            if item.lower() not in Dictionaries.stopwords and \
                    (entity.lower() in item.lower() or item.lower() in entity.lower()):
                ctx_0.append(item)
            else:
                ctx_1.append(item)
        if not ctx_0:
            ent_keyp_with_dis[0] += ctx_0
            ent_keyp_with_dis[1] += ctx_1
    ent_keyp_with_dis_no_dup[0] = set(ent_keyp_with_dis[0])
    ent_keyp_with_dis_no_dup[1] = set(ent_keyp_with_dis[1])
    return ent_keyp_with_dis_no_dup