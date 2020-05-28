import numpy as np
import sys
import csv
import logging_config
from sklearn import preprocessing
import result_folder
import os

result_path = result_folder.__path__[0]
log = logging_config.get_logger()


def read_data_file_csv(filename):
    # FIELD FORMAT: id, other fields, label:1/0
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            data.append(line)
    print('.. Read the data file {} with {} records.'.format(filename, len(data)))
    return data


def delete_features(data, del_opt):
    data = np.delete(data, del_opt, 1)
    return data


def standardization_feature(data):
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)
    return data


def group_men_ent_pairs(data_dev_):
    data_dev = []
    feaid = 0
    while feaid < len(data_dev_):
        feavecs = [data_dev_[feaid]]
        # positive case
        feaid += 1
        while feaid < len(data_dev_) and data_dev_[feaid][-1] == 0:
            feavecs.append(data_dev_[feaid])
            feaid += 1
        feavecs = np.array(feavecs, dtype=np.float)
        data_dev.append(feavecs)
    return data_dev


def predict_idx_linear_combine(data_dev, w_1, w_2):
    scores = data_dev[:, [0]] * w_1 + data_dev[:, [1]] * w_2
    return scores


def test_acc(data_dev, w_1, w_2, res_path):
    matches = 0
    matched_ids = []
    un_matched_ids = []
    # matched_ids = []
    for men_data in data_dev:
        query_id = men_data[0, 0]
        X = men_data[:, 1:-1]
        try:
            X = standardization_feature(X)
        except:
            print(X)
        prediction = predict_idx_linear_combine(X, w_1, w_2)
        #     print(prediction)
        if max(prediction) == prediction[0]:
            matched_ids.append(query_id)
            matches += 1
        else:
            with open(res_path, 'a') as f:
                for item in prediction:
                    f.write("%s\t%s\n" % (query_id, item[0]))
                f.write("\n")
            un_matched_ids.append(query_id)
    # print("LR classes: {}".format(classifier_LR.classes_))
    print("The accuracy is {}".format(matches * 1.0 / len(data_dev)))
    # print("The weights are {}".format(classifier_LR.coef_))
    print("The number of matches is {}, and the number of mentions is {}".format(matches, len(data_dev)))
    with open(res_path, 'a') as f:
        f.write("The accuracy is {}\n".format(matches * 1.0 / len(data_dev)))
        f.write("The number of matches is {}, and the number of mentions is {}\n".format(matches, len(data_dev)))


if __name__ == '__main__':
    filename = './features.csv'
    res_filename = 'idf_entropy_0.1-0.9.result'
    # res_filename = 'idf_entropy_0.1-0.9.all.result'
    # res_filename = 'idf_entropy_0.5-0.5.all.result'
    # res_filename = 'idf_entropy_0.9-0.1.all.result'
    # res_filename = 'idf_entropy_0.9-0.1.all.result'
    # res_filename = 'idf_entropy_0.9-0.1.result'
    # res_filename = 'idf_entropy_0.5-0.5.result'
    # res_filename = 'idf_entropy_0-1.all.result'
    # res_filename = 'idf_entropy_-0.1-1.1.result'
    # res_filename = 'idf_entropy_-0.5-1.5.result'
    # res_filename = 'idf_entropy_-1-2.result'
    # res_filename = 'w_idf_w_entropy_0.9-0.1.all.result'
    # res_filename = 'w_idf_w_entropy_0.5-0.5.all.result'
    # res_filename = 'w_idf_w_entropy_0.1-0.9.all.result'
    # res_filename = 'w_idf_w_entropy_0.2-0.8.result'
    # res_filename = 'w_idf_w_entropy_-0.5-1.5.result'
    # res_filename = 'w_idf_w_entropy_0.9-0.1.result'
    # res_filename = 'w_idf_w_entropy_0-1.result'
    # res_filename = 'w_idf_w_entropy_0.5-0.5.result'
    # res_filename = 'tf_idf_tf_entropy_0.1-0.9.all.result'
    # res_filename = 'tf_idf_tf_entropy_0.9-0.1.all.result'
    # res_filename = 'tf_idf_tf_entropy_1.5--0.5.result'
    # res_filename = 'tf_idf_entropy_0.5-0.5.result'
    # res_filename = 'w_tfidf_w_tf_entropy_0.1-0.9.result'
    # res_filename = 'w_tfidf_w_tf_entropy_0.9-0.1.result'
    # res_filename = 'w_tfidf_w_tf_entropy_0.5-0.5.result'
    # res_filename = 'tfidf_entropy_0.9-0.1.all.result'
    # res_filename = 'tfidf_entropy_0.1-0.9.all.result'
    # res_filename = 'tfidf_entropy_0.5-0.5.all.result'
    # res_filename = 'tfidf_entropy_0.5-0.5.result'
    # res_filename = 'tfidf_entropy_1.5--0.5.all.result'
    # res_filename = 'tfidf_entropy_1--0.all.result'
    res_file_path = os.path.join(result_path, res_filename)
    data = read_data_file_csv(filename)
    # data:  fea = [q_id, etp_sum, w_etp_sum, max_etp, idf_sum, w_idf_sum, max_idf, tf_idf_sum, w_tf_idf_sum,
    # max_tf_idf, tf_entropy_sum, w_tf_entropy_sum, max_tf_entropy, label]
    data = np.array(data, dtype=np.float)
    delete_opetion = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12]  # only q_id, summation of entropy and idf, label are left
    # delete_opetion = [1, 3, 4, 6, 7, 8, 9, 10, 11, 12]  # only q_id, weighted summation of entropy and idf, label are left
    # delete_opetion = [1, 2, 3, 4, 5, 6, 8, 9, 11, 12]  # only q_id, summation of tf*entropy and tf*idf, label are left
    # delete_opetion = [1, 2, 3, 4, 5, 6, 7, 9, 10, 12]  # only q_id, weighted summation of tf*entropy and tf*idf, label are left
    # delete_opetion = [2, 3, 4, 5, 6, 8, 9, 10, 11, 12]  # only q_id, summation of entropy and tf*idf, label are left
    data_dev = delete_features(data, delete_opetion)
    data_dev = group_men_ent_pairs(data_dev)
    entropy_weight = 0.9
    idf_weight = 1 - entropy_weight
    test_acc(data_dev, idf_weight, entropy_weight, res_file_path)
