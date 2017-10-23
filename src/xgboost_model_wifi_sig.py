#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-19, 22:37

@Description:

@Update Date: 17-10-19, 22:37
"""

from util import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import xgboost as xgb
import argparse
import os
parser = argparse.ArgumentParser()
parser.add_argument('--path', required=True, type=str,  help='the path to save the features')

opt = parser.parse_args()
if os.path.exists(os.path.join("../result", opt.path)) == False:
    os.mkdir(os.path.join("../result", opt.path))

def main(offline):
    model_name = "xgboost_wifi_sig"
    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    mall_ids = shop_info.mall_id.unique()
    all_predict = {}
    row_ids_or_true = {}
    # mall_ids = ["m_690"]
    for _index, mall_id in enumerate(mall_ids):
        print "train: ", mall_id, " {}/{}".format(_index, len(mall_ids))
        shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
        num_class = len(shops)
        df, train_cache, test_cache = get_wifi_cache(mall_id)
        train_matrix = train_cache[2]
        train_matrix = np.tile(-train_matrix.max(axis=1,keepdims=True),(1,train_matrix.shape[1])) + train_matrix
        # print(train_matrix.max(axis=1,keepdims=True))
        # exit()
        test_matrix = test_cache[2]

        # 将wifi 信号加上每个sample的最大wifi信号， 屏蔽个体之间接收wifi信号的差异
        train_matrix = np.tile(-train_matrix.max(axis=1, keepdims=True), (1, train_matrix.shape[1])) + train_matrix
        test_matrix = np.tile(-test_matrix.max(axis=1, keepdims=True), (1, test_matrix.shape[1])) + test_matrix

        scala = 1
        pca = PCA(n_components=int(num_class * scala)).fit(train_matrix)
        train_matrix = pca.transform(train_matrix)
        test_matrix = pca.transform(test_matrix)


        test = xgb.DMatrix(test_matrix)

        train = train_all[train_all.mall_id == mall_id]
        test_index = test_cache[0]
        label_encoder = LabelEncoder().fit(shops)
        y = label_encoder.transform(train.shop_id)

        if offline:
            train_matrix, test_matrix, y, test_y = train_test_split(train_matrix, y, test_size=0.1)
        train_x, valid_x, train_y, valid_y = train_test_split(train_matrix, y)

        train = xgb.DMatrix(train_x, label=train_y)
        valid = xgb.DMatrix(valid_x, label=valid_y)
        evals = [(train, "train"), (valid, "valid")]

        if offline:
            test = xgb.DMatrix(test_matrix)

        print "num_class", num_class
        # 模型参数
        objective = "multi:softmax"
        eval_metric = "merror"
        eta = 0.02
        max_depth = 10
        subsample = 0.8
        colsample_bytree = 0.8
        min_child_weight = 5
        gamma = 1
        alpha = 0
        _lambda = 0
        n_round = 200
        early_stop_rounds = 6

        print "train", mall_id
        bst = xgb.train({
            "objective": objective,
            "eval_metric": eval_metric,
            "eta": eta,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "min_child_weight": min_child_weight,
            "gamma": gamma,
            "alpha": alpha,
            "lambda": _lambda,
            "num_class": num_class,
            "silent": 1,
        },
                train,
                n_round,
                evals=evals,
                early_stopping_rounds=early_stop_rounds)

        predict = bst.predict(test, ntree_limit=bst.best_iteration).astype(int)
        predict = label_encoder.inverse_transform(predict)
        all_predict[mall_id] = predict
        if offline:
            test_y = label_encoder.inverse_transform(test_y)
            row_ids_or_true[mall_id] = test_y
        else:
            row_ids_or_true[mall_id] = test_all[np.in1d(test_all.index, test_index)].row_id.values

    if offline:
        result = {}
        for _mall_id in mall_ids:
            _acc = acc(all_predict[_mall_id], row_ids_or_true[_mall_id])
            print _mall_id + "'s acc is", _acc
            result[_mall_id] = _acc
        all_rowid = row_ids_or_true.values()
        all_predict = np.concatenate(all_predict.values())
        all_true = np.concatenate(all_rowid)
        _acc = acc(all_predict, all_true)
        print "all acc is", _acc
        result["all_acc"] = _acc
        path = os.path.join("../result", opt.path, "{}_f{}_eta{}_md{}_ss{}_csb{}_mcw{}_ga{}_al{}_la{}_es{}".format(model_name,
                                                                                                 "num_class_{}".format(
                                                                                                         scala),
                                                                                                 eta, max_depth,
                                                                                                 subsample,
                                                                                                 colsample_bytree,
                                                                                                 min_child_weight,
                                                                                                 gamma,
                                                                                                 alpha, _lambda,
                                                                                                 early_stop_rounds))

        save_acc(result, path, None)

    else:
        all_rowid = np.concatenate(row_ids_or_true.values())
        all_predict = np.concatenate(all_predict.values())
        result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
        result.sort_values(by="row_id", inplace=True)
        path = "../result/online/{}_f{}_eta{}_md{}_ss{}_csb{}_mcw{}_ga{}_al{}_la{}_es{}".format(model_name,
                                                                                                "num_class_{}".format(
                                                                                                        scala),
                                                                                                eta, max_depth,
                                                                                                subsample,
                                                                                                colsample_bytree,
                                                                                                min_child_weight, gamma,
                                                                                                alpha, _lambda,
                                                                                                early_stop_rounds)
        save_result(result, path, None)


if __name__ == '__main__':
    main(offline=True)
