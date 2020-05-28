import argparse
import json
import os
import pickle
from datetime import datetime

import numpy as np
import psycopg2
import redis
import xgboost as xgb
from psycopg2.sql import Identifier, SQL

conn = redis.Redis(
    password='3oSYTdtZjsuSigRWLcG6VJt9gm4IMvYjQiqsSuGcAc-U4gMNpWGERAevXi9_SHNrn19piz7bBJG0iTLgx7DvknLHTECcHYrqmWb2rsuCWs89svKmhKDD_aMYaXq8IhSeg_89ooPZb0AqLRyR1-fa1zVjrh2UuV0sWFGSk5SjtW0',
    host='pangu', port=6379, decode_responses=True)
conn_psql_kongzi2 = psycopg2.connect(database='aida', user='zding', password='dingzishuo', host='localhost',
                                     port=5432)
conn_psql = psycopg2.connect(database='zding', user='zding', password='dingzishuo', host='localhost',
                             port=5432)
model_dir_path = '../model-training/new_models_14_Aug'


def get_feature_results(exp_id, d_t, data_source='aida_conll'):
    print('Getting feature: %s' % exp_id)
    fea_res = conn.lrange('result:::::' + str(exp_id) + ':::::' + data_source + ':::::' + str(d_t), 0, -1)
    fea_vecs = [json.loads(res) for res in fea_res]
    return fea_vecs


def get_dataset_info(exp_id, d_t, data_source='aida_conll', can_size=50):
    valid_mens_size = conn.scard('valid_qry_ids:::::' + str(exp_id) + ':::::' + data_source + ':::::' + str(d_t))
    no_g_can_info_size = conn.scard(
        'no_g_candidate_info:::::' + str(exp_id) + ':::::' + data_source + ':::::' + str(d_t))
    missed_g_candidate_size = conn.scard('missed_g_candidate:::::' + str(exp_id) + ':::::' + data_source +
                                         ':::::' + str(d_t) + ':::::' + str(can_size))
    no_candidate_size = conn.scard('no_candidate:::::' + str(exp_id) + ':::::' + data_source + ':::::' + str(d_t))
    return valid_mens_size, no_candidate_size, no_g_can_info_size, missed_g_candidate_size


def fetch_all_features(exp_id, data_type, data_source='aida_conll'):
    res_feas = get_feature_results(exp_id, data_type, data_source)
    res_feas_ids = [[res[0].strip('(').split(', ')[0]] + res[1:-1] + [res[-1]] for res in res_feas]
    res_feas_ids = np.array(res_feas_ids, dtype=np.float64)
    return res_feas_ids


def fetch_all_features_delete_max_prior(exp_id, data_type, data_source='aida_conll'):
    res_feas = get_feature_results(exp_id, data_type, data_source)
    res_feas_ids = [[res[0].strip('(').split(', ')[0]] + res[1:2] + [res[2] if res[2] != 0 else res[1]] + res[3:] for
                    res in res_feas]
    res_feas_ids = np.array(res_feas_ids, dtype=np.float64)
    return res_feas_ids


def trans_data(data):
    d_np = data[:, 1:-1]
    # print(d_np)
    d_labels = data[:, -1]
    # print(d_labels)
    idxs = np.where(d_labels == 1)[0]
    d_groups = np.append(np.delete(idxs, 0), len(d_labels)) - idxs
    xgb_data = xgb.DMatrix(data=d_np, label=d_labels)
    xgb_data.set_group(d_groups)
    return xgb_data


def combine_features(original_feas, new_features):
    men_id_feas_dict = defaultdict(list)
    print("Building idx for new features...")
    for fea in new_features:
        men_id_feas_dict[fea[0]].append(fea)
    # for k, v in men_id_feas_dict.items():
    #    print(k)
    #    print(v)
    comb_feas = []
    pre_men_id = 0
    print("Combine original and new features...")
    for fea_idx, fea in enumerate(original_feas):
        # print('Processing: [%d] %s' % (fea_idx, fea))
        if pre_men_id == fea[0]:
            # print('skip')
            continue
        else:
            pre_men_id = fea[0]
            fea_size = len(men_id_feas_dict[fea[0]])
            res = np.append(original_feas[fea_idx: fea_idx + fea_size, :-1], np.array(men_id_feas_dict[fea[0]])[:, 1:],
                            axis=1)
            # print('res:', res)
            comb_feas.append(res)
    return np.concatenate(comb_feas, axis=0)


def evalerror(preds, dt, d_tal_size):
    d_l = dt.get_label()
    idxs = np.where(d_l == 1)[0]
    d_groups = np.append(np.delete(idxs, 0), len(d_l)) - idxs
    matched_ids = []
    q_id = 0
    for x in d_groups:
        pre_res = preds[q_id: x + q_id]
        if (preds[q_id] == max(pre_res)):
            if len([x for x in pre_res if x == preds[q_id]]) == 1:
                matched_ids.append(q_id)
        q_id += x
    precision = float(len(matched_ids)) / len(d_groups)
    recall = float(len(matched_ids)) / d_tal_size
    f1 = 2 * precision * recall / (precision + recall)
    return len(matched_ids), precision, recall, f1


def evalerror_detail_log(preds, dt, d_tal_size):
    d_l = dt.get_label()
    idxs = np.where(d_l == 1)[0]
    d_groups = np.append(np.delete(idxs, 0), len(d_l)) - idxs
    correct_results = {}
    wrong_results = {}
    duplicates_results = {}
    group_info = {}
    matched_ids = []
    q_id = 0
    for x in d_groups:
        pre_res = preds[q_id: x + q_id]
        if (preds[q_id] == max(pre_res)):
            correct_results[q_id] = pre_res
            if len([x for x in pre_res if x == preds[q_id]]) == 1:
                matched_ids.append(q_id)
            else:
                duplicates_results[q_id] = pre_res
        else:
            wrong_results[q_id] = pre_res
        q_id += x
    precision = float(len(matched_ids)) / len(d_groups)
    recall = float(len(matched_ids)) / d_tal_size
    f1 = 2 * precision * recall / (precision + recall)
    return len(matched_ids), precision, recall, f1, correct_results, wrong_results, duplicates_results


from collections import defaultdict
from ast import literal_eval


def get_groups_results(preds, dt, res_features, top_k=None):
    d_l = dt.get_label()
    idxs = np.where(d_l == 1)[0]
    d_groups = np.append(np.delete(idxs, 0), len(d_l)) - idxs
    correct_res_groups = []
    wrong_res_groups = []
    dup_res_groups = []
    top_k_indices = [] if top_k is not None else None
    q_id = 0
    for x in d_groups:
        pre_res = preds[q_id: x + q_id]
        pre_res_feas = res_features[q_id: x + q_id]
        pred_q_id, pred_ent = literal_eval(res_features[q_id + np.argmax(pre_res)][0])

        if (preds[q_id] == max(pre_res)):
            correct_res_groups.append([pred_q_id, pred_ent])
        else:
            wrong_res_groups.append([pred_q_id, pred_ent])

        if top_k is not None:  # save indices of top-k scores in each group
            for i, score in sorted(enumerate(pre_res), key=lambda x: x[1], reverse=True)[:top_k]:
                # print(i, score)
                top_k_indices.append(i + q_id)

        q_id += x

    return correct_res_groups, wrong_res_groups, top_k_indices


def fetch_inlinks_by_ent(ent):
    cur = conn_psql_kongzi2.cursor()
    sql = "SELECT _id FROM wikipedia_links_2014 WHERE target=%s;"
    cur.execute(sql, (ent,))
    rows = cur.fetchall()
    cur.close()
    return rows


# fetch an entity's outlinks with duplicates
def fetch_outlinks_by_ent(ent):
    cur = conn_psql_kongzi2.cursor()
    sql = "SELECT target FROM wikipedia_links_2014 WHERE _id=%s;"
    cur.execute(sql, (ent,))
    rows = cur.fetchall()
    cur.close()
    return rows


def fetch_entity_by_mention_emnlp17(mention):
    # print(mention)
    cur = conn_psql_kongzi2.cursor()
    # do a PostgreSQL join to select the entity namestring from the tables dictionary and entity_ids
    sql = "SELECT entity, prior FROM men_ent_dict_emnlp2017 WHERE men_ent_dict_emnlp2017.mention = (E\'%s\') ORDER BY prior DESC;"
    cur.execute(sql % mention.replace("'", "\\'"))
    rows = cur.fetchall()
    cur.close()
    return rows


def fetch_inlinks_redis(ent, link_type='inlinks'):
    inlinks = conn.hmget(link_type, ent)[0]
    return json.loads(inlinks) if inlinks else []


def has_inlinks_redis(ent, link_type='inlinks'):
    inlinks = conn.hexists(link_type, ent)
    return inlinks


def save_inlinks_redis(ent, inlinks, link_type='inlinks'):
    conn.hset(link_type, ent, json.dumps(inlinks))


def fetch_outlinks_redis(ent, link_type='outlinks'):
    outlinks = conn.hmget(link_type, ent)[0]
    return json.loads(outlinks) if outlinks else []


def has_outlinks_redis(ent, link_type='outlinks'):
    inlinks = conn.hexists(link_type, ent)
    return inlinks


def save_outlinks_redis(ent, outlinks, link_type='outlinks'):
    conn.hset(link_type, ent, json.dumps(outlinks))


def check_links_between_ents(ent_1, ent_2, bidirection=False):
    wiki_pre_str = 'en.wikipedia.org/wiki/'
    inlinks_ent_1 = fetch_inlinks_redis(ent_1, link_type='inlinks')
    if not inlinks_ent_1 and not has_inlinks_redis(ent_1):
        print("PostgreSQL: fetching inlinks for entity {}...".format(ent_1))
        wiki_ents_1 = wiki_pre_str + ent_1
        inlinks_ent_1_db = fetch_inlinks_by_ent(wiki_ents_1)
        inlinks_ent_1 = [x[0].replace(wiki_pre_str, '') for x in inlinks_ent_1_db]
        print("Redis: caching inlinks for entity {}...".format(ent_1))
        save_inlinks_redis(ent_1, inlinks_ent_1)
    inlinks_ent_2 = fetch_inlinks_redis(ent_2, link_type='inlinks')
    if not inlinks_ent_2 and not has_inlinks_redis(ent_2):
        wiki_ents_2 = wiki_pre_str + ent_2
        print("PostgreSQL: fetching inlinks for entity {}...".format(ent_2))
        inlinks_ent_2_db = fetch_inlinks_by_ent(wiki_ents_2)
        inlinks_ent_2 = [x[0].replace(wiki_pre_str, '') for x in inlinks_ent_2_db]
        print("Redis: caching inlinks for entity {}...".format(ent_2))
        save_inlinks_redis(ent_2, inlinks_ent_2)
    return (ent_1 in inlinks_ent_2 or ent_2 in inlinks_ent_1) if not bidirection else (
            ent_1 in inlinks_ent_2 and ent_2 in inlinks_ent_1)


def get_links_by_ent(ent, link_type='inlinks'):
    wiki_pre_str = 'en.wikipedia.org/wiki/'
    if link_type == 'inlinks':
        inlinks_ent = fetch_inlinks_redis(ent, link_type='inlinks')
        if not inlinks_ent and not has_inlinks_redis(ent):
            #             print("PostgreSQL: fetching inlinks for entity {}...".format(ent))
            wiki_ents = wiki_pre_str + ent
            inlinks_ent_db = fetch_inlinks_by_ent(wiki_ents)
            inlinks_ent = [x[0].replace(wiki_pre_str, '') for x in inlinks_ent_db]
            print("Redis: caching inlinks for entity {}...".format(ent))
            save_inlinks_redis(ent, inlinks_ent)
        return inlinks_ent
    if link_type == 'outlinks':
        outlinks_ent = fetch_outlinks_redis(ent)
        if not outlinks_ent and not has_outlinks_redis(ent):
            #             print("PostgreSQL: fetching outlinks for entity {}...".format(ent))
            wiki_ents = wiki_pre_str + ent
            outlinks_ent_db = fetch_outlinks_by_ent(wiki_ents)
            outlinks_ent = [x[0].replace(wiki_pre_str, '') for x in outlinks_ent_db]
            print("Redis: caching outlinks for entity {}...".format(ent))
            save_outlinks_redis(ent, outlinks_ent)
        return outlinks_ent


def fetch_ents_by_doc_redis(doc_id):
    id_ents = conn.hmget('doc-predicted-ents-coref-new', doc_id)[0]
    return json.loads(id_ents) if id_ents else []


## Normalized Google Distance
import math


def ngd_similarity(ents_s, ents_t, index_size=6274625):
    ent_sets_s = set(ents_s)
    ent_sets_t = set(ents_t)
    min_links, max_links = min(len(ent_sets_s), len(ent_sets_t)), max(len(ent_sets_s), len(ent_sets_t))
    com_links = len(ent_sets_s & ent_sets_t)
    if min_links and max_links and com_links:
        return 1 - (math.log(max_links) - math.log(com_links)) / (math.log(index_size) - math.log(min_links))
    else:
        return 0


# PMI
def pmi_similarity(ents_s, ents_t, index_size=6274625, normalize=False):
    ent_sets_s = set(ents_s)
    ent_sets_t = set(ents_t)
    s_links, t_links = len(ent_sets_s), len(ent_sets_t)
    com_links = len(ent_sets_s & ent_sets_t)
    p_s = s_links / index_size
    p_t = t_links / index_size
    p_c = com_links / index_size
    print(p_s, p_t, p_c)
    if p_s and p_t and p_c:
        return p_c / (p_s * p_t) if not normalize else p_c / (p_s * p_t) / min(1 / p_s, 1 / p_t)
    else:
        return 0


def save_model(model, name):
    if not os.path.exists(model_dir_path):
        os.makedirs(model_dir_path)
    model_path = os.path.join(model_dir_path, '%s.mdl' % name)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)


def load_model(name):
    model_path = os.path.join(model_dir_path, '%s.mdl' % name)
    with open(model_path, 'rb') as f:
        return pickle.load(f)


def get_total_mentions(data_source, data_type) -> int:
    with conn_psql.cursor() as cur:
        sql = SQL("select count(*) from {} where annotation != 'NIL' and annotation !='none' and type=%s").format(
            Identifier(data_source))
        cur.execute(sql, (data_type,))
        return cur.fetchone()[0]


def process(process_name, test_set, test_total,
            n_estimators, max_depths, test_filter=None,
            eval_func=evalerror_detail_log):
    if test_filter is not None:
        dtest_xgboost = trans_data(test_set[test_filter])
    else:
        dtest_xgboost = trans_data(test_set)

    for x in n_estimators:
        num_round = x
        for dep in max_depths:
            model_name = '%d_%d_%s' % (num_round, dep, process_name)
            print(datetime.now(), 'Loading model: %s' % model_name)
            bst = load_model(model_name)

            print(datetime.now(), 'Start evaluation')
            preds = bst.predict(dtest_xgboost)
            a = eval_func(preds, dtest_xgboost, test_total)
            print("n_estimators: {}, max_depth: {}, precision: {}, recall: {}, f1: {}, corr_num: {}"
                  .format(num_round, dep, a[1], a[2], a[3], a[0]))
            print(datetime.now(), 'Evaluation finished')


def fetch_q_ids_docs(data_source):
    cur = conn_psql.cursor()
    sql = "SELECT id, doc_id FROM %s WHERE annotation != 'NIL' and annotation != 'none';" % data_source
    cur.execute(sql)
    row = cur.fetchall()
    cur.close()
    return dict(row)


def save_local_model_predictions(model, data_source, data_type, d_test, raw_test, docs_dict, top_k=None):
    d_test_xgboost = trans_data(d_test)
    preds_test = model.predict(d_test_xgboost)

    correct_test, wrong_test, top_k_indices_test = get_groups_results(preds_test, d_test_xgboost, raw_test, top_k)
    res_all_test = correct_test + wrong_test

    print('Number of groups:', len(res_all_test))

    doc_id_q_ent_lists_dict = defaultdict(list)
    for q_ent in res_all_test:
        doc_id_q_ent_lists_dict[docs_dict[q_ent[0]]].append(q_ent)

    for key, vals in doc_id_q_ent_lists_dict.items():
        conn.hset('doc-predicted-ents:%s:%s' % (data_source, data_type), key, json.dumps(vals))

    if top_k is not None:
        conn.hset('doc-predicted-ents-top-k:%s:%s' % (data_source, data_type), 'test', repr(top_k_indices_test))


def save_model_predictions_extended(model, data_source, data_type, d_test, res_features, docs_dict, top_k=1):
    dt = trans_data(d_test)
    preds = model.predict(dt)

    d_l = dt.get_label()
    idxs = np.where(d_l == 1)[0]
    d_groups = np.append(np.delete(idxs, 0), len(d_l)) - idxs

    all_group_results = []
    q_id = 0
    for x in d_groups:
        pre_res = preds[q_id: x + q_id]
        pre_ranking = sorted(enumerate(pre_res), key=lambda x: x[1], reverse=True)[:top_k]
        group_results = []
        pred_q_id = None
        for idx, score in pre_ranking:
            pred_q_id, pred_ent = literal_eval(res_features[q_id + idx][0])
            group_results.append((pred_ent, float(score)))
        all_group_results.append((pred_q_id, group_results))
        q_id += x

    print('Number of groups:', len(all_group_results))

    doc_id_q_ent_lists_dict = defaultdict(list)
    for q_id_group_result in all_group_results:
        q_id, group_results = q_id_group_result
        doc_id_q_ent_lists_dict[docs_dict[q_id]].append(q_id_group_result)
        conn.hset('mention-candidate-filter:%s:%s' % (data_source, data_type), q_id,  json.dumps(group_results))

    for key, vals in doc_id_q_ent_lists_dict.items():
        conn.hset('doc-predicted-ents-ext:%s:%s' % (data_source, data_type), key, json.dumps(vals))



def save_ground_truths_as_local_model_predictions(raw_data, docs_dict, data_source, data_type):
    predictions = []
    for row in raw_data:
        mention_id, entity = eval(row[0])
        if row[-1]:
            predictions.append((mention_id, entity))

    print('Number of groups:', len(predictions))

    doc_id_q_ent_lists_dict = defaultdict(list)
    for q_ent in predictions:
        doc_id_q_ent_lists_dict[docs_dict[q_ent[0]]].append(q_ent)

    for key, vals in doc_id_q_ent_lists_dict.items():
        conn.hset('doc-predicted-ents:%s:%s' % (data_source, data_type), key, json.dumps(vals))


def get_true_labels(data_set):
    return data_set[:, -1].nonzero()[0].size


def main(data_source, data_type, iteration, model_suffix='', feature_suffix='', save_format='default', top_k=3):
    print(datetime.now(), 'Loading meta info and local features...')
    docs_dict = fetch_q_ids_docs(data_source)
    d_ctx_raw = get_feature_results('basic_fea_ctx', data_type, data_source)
    d_ctx = fetch_all_features('basic_fea_ctx', data_type, data_source)
    d_coref = fetch_all_features('basic_fea_coref', data_type, data_source)

    # d_total = get_total_mentions(data_source, data_type)
    # d_ctx_true_labels = get_true_labels(d_ctx)

    # print(d_ctx.shape)
    # print(d_coref.shape)
    # print(d_ctx_true_labels)
    # print('upper bound', d_ctx_true_labels / d_total)

    d_ctx_coref = combine_features(d_ctx, d_coref)

    if iteration == 0:
        model_name = '4900_6_ctx_coref'  # local model
        coh_feature_name = None
    elif iteration == 1:
        model_name = '4900_6_ctx_coref_coh%s' % model_suffix  # initial global model
        coh_feature_name = 'basic_fea_coh%s' % feature_suffix
    elif iteration == 2:
        model_name = '4900_6_ctx_coref_coh%s_global' % model_suffix  # 1st iterated global model
        coh_feature_name = 'basic_fea_coh%s_global' % feature_suffix
    else:
        model_name = '4900_6_ctx_coref_coh%s_global%d' % (model_suffix, iteration - 1)  # next iterated global models
        coh_feature_name = 'basic_fea_coh%s_global%d' % (feature_suffix, iteration - 1)

    if coh_feature_name:
        print('Loading coh features from %s' % coh_feature_name)
        d_coh = fetch_all_features(coh_feature_name, data_type, data_source)
        # print(d_coh.shape)
        d_all_features = combine_features(d_ctx_coref, d_coh)
        # print(d_ctx_coref_coh.shape)
    else:
        d_all_features = d_ctx_coref

    print(datetime.now(), 'Loading model: %s...' % model_name)
    model = load_model(model_name)
    print(datetime.now(), 'Saving predictions with format: %s' % save_format)
    if save_format == 'default':
        save_local_model_predictions(model, data_source, data_type, d_all_features, d_ctx_raw, docs_dict)
    elif save_format == 'ext':
        save_model_predictions_extended(model, data_source, data_type, d_all_features, d_ctx_raw, docs_dict, top_k)
    else:
        raise ValueError('invalid save format: %s' % save_format)
    print(datetime.now(), 'Done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Save predictions of a local/global model in an iteration step')
    parser.add_argument('data_source')
    parser.add_argument('data_type')
    parser.add_argument('iteration', type=int, default=0)
    parser.add_argument('-m', '--model_suffix', default='')
    parser.add_argument('-f', '--feature_suffix', default='')
    parser.add_argument('-t', '--top_k', type=int, default=3)
    parser.add_argument('-s', '--save_format', default='default')
    args = parser.parse_args()

    main(args.data_source, args.data_type, args.iteration, args.model_suffix, args.feature_suffix,
         args.save_format, args.top_k)
