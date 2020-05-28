from const_vars.const_vars_abc import Dictionaries
import logging_config
import result_folder
result_path = result_folder.__path__[0]

log = logging_config.get_logger()


def normalize_form(mention):
    return mention if len(mention) < 4 else str(mention).upper()


def escape_entity(entity):
    return str.encode(entity).decode('unicode-escape')


def check_recall_redis(candidate_with_prior_list, pair, can_size=50):
    if len(candidate_with_prior_list) == 0:
        Dictionaries.redis_db_obj.save_set_redis('no_candidate:::::' + Dictionaries.experiment_id + ":::::" +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type, str(pair[0]))
        return False
    if escape_entity(pair[2]) not in candidate_with_prior_list:
        Dictionaries.redis_db_obj.save_set_redis('missed_g_candidate:::::' + Dictionaries.experiment_id + ":::::" +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type + ":::::" + str(can_size), str(pair[0]))
        return False
    if not Dictionaries.redis_db_obj.is_processed_ents_redis(Dictionaries.ent_repo_id, escape_entity(pair[2])):
        Dictionaries.redis_db_obj.save_set_redis('no_g_candidate_info:::::' + Dictionaries.experiment_id + ":::::" +
                                                 Dictionaries.dataset_source + ':::::' +
                                                 Dictionaries.dataset_type, str(pair[0]))
        return True
    else:
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


def get_valid_mention_entity_pairs(id_start, id_end, gen_all, dataset_type, can_size):
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
        log.info("positive pairs generation {}, {}, {}".format(str(pair[0]), pair[1], escape_entity(pair[2])))

        log.info("Get the candidate entity list and prior for mention: {}".format(pair[1]))
        candidate_with_prior_db = Dictionaries.prior_db_conn_obj.fetch_entity_by_mention_size(normalize_form(pair[1]),
                                                                                              can_size)
        log.info("Finished Generation of the candidate entities, {} candidates are generated".format(len(
            candidate_with_prior_db)))
        candidate_with_prior_list = dict(candidate_with_prior_db)
        if not check_recall_redis(candidate_with_prior_list, pair):
            continue
        # one positive case, [query_id, prior, candidate entity, mention, doc_id, 1/0 (pos/neg)]
        positive_case = [pair[0], candidate_with_prior_list[escape_entity(pair[2])], escape_entity(pair[2]), pair[1],
                         pair[-1], 1]
        mep.append(positive_case)

        negative_entity_list = [candidate[0] for candidate in candidate_with_prior_db]

        negative_entity_list.remove(escape_entity(pair[2]))
        check_processed_neg_ent_redis(pair[0], negative_entity_list)
        for negative_entity in negative_entity_list:
            log.info("Negative pairs generation %s, %s, %s\r" % (str(pair[0]), pair[1], negative_entity))
            negative_case = [pair[0], candidate_with_prior_list[negative_entity], negative_entity, pair[1], pair[-1], 0]
            mep.append(negative_case)
        men_ent_pairs.append(mep)
    return men_ent_pairs


if __name__ == '__main__':
    Dictionaries.init_dictionaries()
    cases = get_valid_mention_entity_pairs()
