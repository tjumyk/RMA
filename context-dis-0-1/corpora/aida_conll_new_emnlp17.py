"""
Fetch cases from the database
"""
from const_vars.const_vars_abc import Dictionaries
from const_vars import const_vars_abc
import logging_config
import result_folder
import json
result_path = result_folder.__path__[0]

log = logging_config.get_logger()


def normalize_form(mention):
    return mention if len(mention) < 4 else str(mention).upper()


def escape_entity(entity):
    return str.encode(entity).decode('unicode-escape')


def check_recall_redis(candidate_with_prior_list, pair, ent, can_size=50):
    if len(candidate_with_prior_list) == 0:
        Dictionaries.redis_db_obj.save_set_redis('no_candidate:::::' + Dictionaries.experiment_id + ":::::" +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type, str(pair[0]))
        return False
    if ent not in candidate_with_prior_list:
        Dictionaries.redis_db_obj.save_set_redis('missed_g_candidate:::::' + Dictionaries.experiment_id + ":::::" +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type + ":::::" + str(can_size), str(pair[0]))
        return False
    if not Dictionaries.redis_db_obj.is_processed_ents_redis(Dictionaries.ent_repo_id, ent):
        Dictionaries.redis_db_obj.save_set_redis('no_pro_ent:::::' + Dictionaries.experiment_id + ":::::" +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type, str(pair[0]))
        return True
    else:
        # if not Dictionaries.redis_db_obj.get_ent_tokens_redis(Dictionaries.ent_repo_id, ent):
        #     Dictionaries.redis_db_obj.save_set_redis(
        #         'no_grd_ent_article:::::' + Dictionaries.experiment_id + ":::::" +
        #         Dictionaries.dataset_source + ':::::' +
        #         Dictionaries.dataset_type, str(pair[0]))
        Dictionaries.redis_db_obj.save_set_redis('valid_qry_ids:::::' + Dictionaries.experiment_id + ':::::' +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type, str(pair[0]))
        return True


def check_processed_neg_ent_redis(query_id, neg_ent_lst):
    n_processed_ents = Dictionaries.redis_db_obj.get_n_processed_ents_redis(Dictionaries.ent_repo_id, neg_ent_lst)
    if n_processed_ents:
        Dictionaries.redis_db_obj.save_set_redis('no_neg_ent_ctx:::::' + Dictionaries.experiment_id + ':::::' +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type,
                                                 [str(query_id) + '::::' + neg_ent for neg_ent in n_processed_ents])


def get_valid_mention_entity_pairs(id_start, id_end, gen_all, dataset_type, can_size=30, find_ground_truths=False, use_candidate_filter=False):
    """
    Generate candidate entities for every non-nil mentions in aida-conll
    :return:[[[]], ...]
    """
    if gen_all:
        log.info("Generating the feature vectors for all the cases...")
        mention_entity_pairs_db = Dictionaries.db_conn_obj.fetch_mention_entity_pairs(Dictionaries.dataset_source,
                                                                                      dataset_type)
    else:
        log.info("Generating the feature vectors for {}, with query id between {} and {}...".format(dataset_type,
                                                                                                    id_start, id_end))
        mention_entity_pairs_db = Dictionaries.db_conn_obj.fetch_mention_entity_pairs_by_ids(
            Dictionaries.dataset_source, Dictionaries.dataset_type, id_start, id_end)
    men_ent_pairs = []

    for pair in mention_entity_pairs_db:

        # a list with one positive case and multiple negative cases
        mep = []
        
        if use_candidate_filter:
            candidate_filter = Dictionaries.redis_db_obj.conn.hget('mention-candidate-filter:%s:%s' % \
                                    (Dictionaries.dataset_source, dataset_type), pair[0])
            if candidate_filter:
                candidate_filter = dict(json.loads(candidate_filter))
            if not candidate_filter:
                continue  # empty candidate list, skip this pair
        else:
            candidate_filter = None
        n_men, candidate_with_prior_db = const_vars_abc.preprocess_mention(pair[1], candidate_filter=candidate_filter)

        wiki_id = escape_entity(pair[2]) if Dictionaries.dataset_source == 'aida_conll' else pair[2]

        n_ent = const_vars_abc.get_redirect_entity(wiki_id)
        log.info("positive pairs generation {}, {} --> {}, {} --> {}".
                 format(str(pair[0]), pair[1], n_men, wiki_id, n_ent))

        if find_ground_truths and candidate_with_prior_db:
            # get redirects of all the candidate entities
            ent_redirects = const_vars_abc.get_redirect_entity_batch([ent for ent, prior in candidate_with_prior_db])
            candidate_with_prior_db = [(ent_redirect, prior) for (ent, prior), ent_redirect 
                                       in zip(candidate_with_prior_db, ent_redirects)]
            
            # save some info about where the ground truth is
            if ent_redirects[0] == n_ent:
                Dictionaries.redis_db_obj.conn.sadd('dict-top-is-ground-truth:%s:%s' 
                                                    % (Dictionaries.dataset_source, dataset_type), pair[0])
            if any(ent == n_ent for ent in ent_redirects):
                Dictionaries.redis_db_obj.conn.sadd('dict-has-ground-truth:%s:%s' 
                                                    % (Dictionaries.dataset_source, dataset_type), pair[0])

        candidate_with_prior_db = \
            candidate_with_prior_db[: min(can_size, len(candidate_with_prior_db))]

        if not find_ground_truths and candidate_with_prior_db: 
            # get redirects of the top candidate entities
            ent_redirects = const_vars_abc.get_redirect_entity_batch([ent for ent, prior in candidate_with_prior_db])
            candidate_with_prior_db = [(ent_redirect, prior) for (ent, prior), ent_redirect
                                       in zip(candidate_with_prior_db, ent_redirects)]
        # the original code for getting redirects
        # candidate_with_prior_db = [(const_vars_abc.get_redirect_entity(x[0]), x[1])
        #                            for x in candidate_with_prior_db]

        candidate_with_prior_list = dict(candidate_with_prior_db)
        if not check_recall_redis(candidate_with_prior_list, pair, n_ent):
            continue
        # one positive case, in the following format
        # [query_id, prior, candidate entity, mention, doc_id, begoffset, 1/0 (pos/neg)]
        positive_case = [pair[0], candidate_with_prior_list[n_ent], n_ent, pair[1], pair[-1],
                         int(pair[4]), 1]
        mep.append(positive_case)

        negative_entity_list = [candidate[0] for candidate in candidate_with_prior_db]

        negative_entity_list.remove(n_ent)
        # check_processed_neg_ent_redis(pair[0], negative_entity_list)
        for negative_entity in negative_entity_list:
            log.info("Negative pairs generation %s, %s, %s\r" % (str(pair[0]), pair[1], negative_entity))
            negative_case = [pair[0], candidate_with_prior_list[negative_entity], negative_entity,
                             pair[1], pair[-1], int(pair[4]), 0]
            mep.append(negative_case)
        men_ent_pairs.append(mep)
    return men_ent_pairs


if __name__ == '__main__':
    Dictionaries.init_dictionaries()
    cases = get_valid_mention_entity_pairs()
