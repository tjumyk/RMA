import logging_config
import argparse
import time
import json
# import const_vars.constant_conll_testb
# import corpora.ace as cop
# import corpora.aquaint_new as cop
# import corpora.aida_conll_new as cop
# import corpora.aida_conll_new_sup as cop
# import corpora.msnbc_new as cop
# import corpora.conll
# import corpora.aida_conll_new_emnlp17 as cop
# import corpora.aida_conll_new_emnlp17_sup as cop
import const_vars.const_vars_abc as c_vars
# import result_folder
# from feature_generator.fea_gen_ctx_dis_1 import process
# from feature_generator.fea_gen_ctx_basic import process
# from feature_generator.fea_gen_ctx_basic_ctx_dis_1 import process
# from feature_generator.fea_gen_token_phrase_idf_entropy import process
# from feature_generator.fea_gen_token_phrase_idf_entropy_basic_poly import process
# from feature_generator.fea_gen_token_coref import process
# from feature_generator.fea_gen_ents_coh import process
#from feature_generator.fea_gen_token_phrase_idf_entropy_basic_poly_emnlp17 import process
# from feature_generator.fea_gen_token_coref_emnlp17 import process
# from feature_generator.fea_gen_ents_coh_emnlp17 import process
log = logging_config.get_logger()

# config_path = database.__path__[0]
# config_path = os.path.join(config_path, 'config.json')
# result_path = result_folder.__path__[0]
#
# with open(config_path, 'r') as f:
#     config = json.load(f)
# threads_num = config['threads_num']
#
#
const_vars_dict = c_vars.Dictionaries


def save_res_2_redis(results):
    fea_ids = []
    fea_vecs = []
    for res in results:
        fea_ids.append(res[0])
        fea_vecs.append(json.dumps(res))
    if fea_ids and fea_vecs:
        const_vars_dict.redis_db_obj.save_list_redis(
            'result:::::' + const_vars_dict.experiment_id + ':::::' +
            const_vars_dict.dataset_source + ':::::' +
            const_vars_dict.dataset_type, fea_vecs)
        const_vars_dict.redis_db_obj.save_set_redis(
            'processed_queries:::::' + const_vars_dict.experiment_id + ':::::' +
            const_vars_dict.dataset_source + ':::::' +
            const_vars_dict.dataset_type, fea_ids)


def main_multiprocessing(id_start, id_end, gen_all, dataset_type, candidate_size):
    s_t = time.time()
    log.info("Generating entity linking queries...")
    cases = cop.get_valid_mention_entity_pairs(id_start, id_end, gen_all, dataset_type, can_size=candidate_size, use_candidate_filter=False)
    log.info("Finished Generation of cases, time cost {}".format(time.time() - s_t))
    try:
        for case_idx, case in enumerate(cases):
            if not case_idx % 50:
                const_vars_dict.redis_db_obj.\
                    save_dict_redis('py-status', const_vars_dict.p_name,
                                    'running. Info:{}, {}, {}, {}, {}, {}/{}'
                                    .format(const_vars_dict.experiment_id,
                                    const_vars_dict.m_start, const_vars_dict.m_end, gen_all,
                                    const_vars_dict.dataset_type, case_idx,
                                    const_vars_dict.m_end - const_vars_dict.m_start))
            results = process(case)
            save_res_2_redis(results)
    except:
        log.exception("Exception occurred")
        raise


if __name__ == '__main__':
    ti = time.time()
    parser = argparse.ArgumentParser("Generate feature vectors for entity linking cases.\n")
    parser.add_argument("x", type=int, help="The beginning index of the id for queries.",
                        default=0)
    parser.add_argument("y", type=int, help="The ending index of the id for queries.", default=0)
    parser.add_argument("s", type=str, help="The source of the dataset, such as conll, ace....",
                        default='conll')
    parser.add_argument("t", type=str, help="Type of the data set: train, testa, or testb.",
                        default='testb')
    parser.add_argument("p_name", type=str, help="Type of the data set: train, testa, or testb.",
                        default='error')
    parser.add_argument("exp_id", type=str, help="Experiment ID.", default='error')
    # parser.add_argument("name", type=str, help="The name of the program.", default=0)
    parser.add_argument("-a", "--all", action="store_true",
                        help="Generate the feature vectors for all the cases.")
    parser.add_argument('-c', '--candidate_size', help='Candidate size', type=int, default=50)
    parser.add_argument('feature_type', help='Feature type', default='basic')
    args = parser.parse_args()

    print('Corpus: %s' % args.s)
    # if args.s == 'ace2004_uiuc':
    #     import corpora.ace as cop
    # elif args.s == 'aquaint_new':
    #     import corpora.aquaint_new as cop
    # elif args.s == 'aida_conll':
    import corpora.aida_conll_new_emnlp17 as cop
    # elif args.s == 'msnbc_new':
    #     import corpora.msnbc_new as cop
    # else:
    #     raise ValueError('invalid data source: %s' % args.s)

    print('Feature type: %s' % args.feature_type)
    if args.feature_type == 'ctx':
        from feature_generator.fea_gen_token_phrase_idf_entropy_basic_poly_emnlp17 import process
    elif args.feature_type == 'ctx_no_poly':
        from feature_generator.fea_gen_token_phrase_idf_entropy import process
    elif args.feature_type == 'ctx_poly':
        from feature_generator.fea_gen_token_phrase_idf_entropy_basic_poly import process
    elif args.feature_type == 'ctx_poly_no_normalized_tf':
        from feature_generator.fea_gen_token_phrase_idf_entropy_basic_poly_no_normalized_tf import process
    elif args.feature_type == 'ctx_simplified':
        from feature_generator.fea_gen_token_phrase_idf_entropy_basic_poly_emnlp17_simplified import process
    elif args.feature_type == 'coref':
        from feature_generator.fea_gen_token_coref_emnlp17 import process    
    elif args.feature_type == 'coref_simplified':
        from feature_generator.fea_gen_token_coref_emnlp17_simplified import process
    elif args.feature_type == 'coh':
        from feature_generator.fea_gen_ents_coh_emnlp17 import process
    elif args.feature_type == 'coh_ext':
        from feature_generator.fea_gen_ents_coh_emnlp17_ext import process
    elif args.feature_type == 'coh_ext2':
        from feature_generator.fea_gen_ents_coh_emnlp17_ext2 import process
    elif args.feature_type == 'coh_ext3':
        from feature_generator.fea_gen_ents_coh_emnlp17_ext3 import process
    elif args.feature_type in {'coh_max_max', 'coh_max_avg', 'coh_avg_avg', 'coh_avg_max',
                               'coh_top3_avg', 'coh_top3_max',
                               'coh_orig-max_max', 'coh_orig-max_avg', 'coh_orig-avg_avg',
                               'coh_orig-avg_max', 'coh_orig-top3_avg', 'coh_orig-top3_max',
                               'coh_max_max-max_avg', 'coh_avg_avg-avg_max',
                               'coh_max_max-avg_avg', 'coh_max_avg-avg_max',
                               'coh_max_max-avg_max', 'coh_max_avg-avg_avg',
                               'coh_top3_avg-top3_max'}:
        from feature_generator.fea_gen_ents_coh_emnlp17_ext_all import process as _process
        def process(cases):
            return _process(cases, args.feature_type[4:], candidate_top_k=3)
    #elif args.feature_type == 'pred_score':
    #    from feature_generator.fea_gen_pred_score import process
    else:
        raise ValueError('invalid feature type: %s' % args.feature_type)

    const_vars_dict.m_start = args.x
    const_vars_dict.m_end = args.y
    # load constant variables from db
    log.info("Loading dictionary...")
    const_vars_dict.init_dictionaries()
    log.info("Finished Loading dictionary.")

    const_vars_dict.dataset_type = args.t
    const_vars_dict.experiment_id = args.exp_id
    # const_vars_dict.ent_repo_id = 'ent-tok-ner-spacy_md-new'
    const_vars_dict.ent_repo_id = 'ent-tok-ner-spacy-30-new'
    const_vars_dict.dataset_source = args.s

    const_vars_dict.init_max_ent_priors()

    # save the program running status into Redis
    const_vars_dict.redis_db_obj.save_dict_redis('py-status', args.p_name,
                                                 'running. Info:{}, {}, {}, {}, {}'
                                                 .format(args.exp_id, args.x, args.y, args.all,
                                                         args.t))
    const_vars_dict.p_name = args.p_name
    # const_vars_dict.set_dataset_type(args.t)
    # const_vars_dict.set_experiment_id(args.exp_id)

    try:
        main_multiprocessing(args.x, args.y, args.all, args.t, args.candidate_size)
        const_vars_dict.redis_db_obj.save_dict_redis('py-status', args.p_name,
                                                     'done -> stopped. Info:{}, {}, {}, {}, {}'
                                                     .format(args.exp_id, args.x, args.y, args.all,
                                                             args.t))
    except Exception as e:
        const_vars_dict.redis_db_obj.save_dict_redis('py-status', args.p_name,
                                                     'exception -> stopped. Info:{}, {}, {}, {}, {}, {}'
                                                     .format(args.exp_id, args.x, args.y, args.all,
                                                             args.t, e))
        log.error(e, exc_info=True)
