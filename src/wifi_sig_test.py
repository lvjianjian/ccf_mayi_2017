#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-19, 21:41

@Description:

@Update Date: 17-10-19, 21:41
"""

from util import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb


def main1(mall_id): #pca
    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
    num_class = len(shops)
    df, train_cache, test_cache = get_wifi_cache(mall_id)
    train_index = train_cache[0]
    train_use_wifi = train_cache[1]
    train_matrix = train_cache[2]
    scala = 1
    train_matrix = PCA(n_components=int(num_class * scala)).fit_transform(train_matrix)
    # train_matrix = train_matrix[:, :300]
    # print df[:300]
    # train_matrix = (train_matrix[:] > -90).astype(int)
    # print train_matrix.sum()

    train = train_all[train_all.mall_id == mall_id]

    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)
    train_x, test_x, train_y, test_y = train_test_split(train_matrix, y)

    train = xgb.DMatrix(train_x, label=train_y)
    test = xgb.DMatrix(test_x, label=test_y)
    evals = [(train, "train"), (test, "valid")]

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
    early_stop_rounds = 10

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


def main2(mall_id): # LDA
    train_all = load_train()
    shop_info = load_shop_info()
    shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
    num_class = len(shops)

    train = train_all[train_all.mall_id == mall_id]
    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)

    df, train_cache, test_cache = get_wifi_cache(mall_id)
    train_matrix = train_cache[2]
    train_x, test_x, train_y, test_y = train_test_split(train_matrix, y)


    scala = 1
    # lda = LinearDiscriminantAnalysis(n_components=num_class).fit(np.concatenate([train_x, valid_x]), np.concatenate([train_y,valid_y]))
    lda = LinearDiscriminantAnalysis(n_components=num_class).fit(train_x, train_y)
    # lda.predict(test_x)
    p = lda.predict(test_x)

    print "LDA", acc(p,test_y)
    # pca = PCA(n_components=int(num_class * scala)).fit(np.concatenate([train_x, valid_x]))
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    train = xgb.DMatrix(train_x, label=train_y)
    valid = xgb.DMatrix(test_x, label=test_y)
    evals = [(train, "train"), (valid, "valid")]

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
    early_stop_rounds = 8

    print "train", mall_id
    params = {
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
    }

    bst = xgb.train(params,
                    train,
                    n_round,
                    evals=evals,
                    early_stopping_rounds=early_stop_rounds)

    p = bst.predict(valid)
    print "lda decomposition xgboost acc", acc(p,test_y)
    # bst = xgb.train(params, n_round=bst.best_iteration )

def main3(mall_id): # KNN
    train_all = load_train()
    shop_info = load_shop_info()
    shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
    num_class = len(shops)

    train = train_all[train_all.mall_id == mall_id]
    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)

    df, train_cache, test_cache = get_wifi_cache(mall_id)
    train_matrix = train_cache[2]
    train_matrix = train_matrix[:, :3000]
    train_x, test_x, train_y, test_y = train_test_split(train_matrix, y)

    knn = KNeighborsClassifier().fit(train_x, train_y)
    p = knn.predict(test_x)
    print "knn acc", acc(p,test_y)
    exit(1)
    scala = 1
    # lda = LinearDiscriminantAnalysis(n_components=num_class).fit(np.concatenate([train_x, valid_x]), np.concatenate([train_y,valid_y]))
    lda = LinearDiscriminantAnalysis(n_components=num_class).fit(train_x, train_y)
    # lda.predict(test_x)
    p = lda.predict(test_x)

    print "LDA", acc(p,test_y)
    # pca = PCA(n_components=int(num_class * scala)).fit(np.concatenate([train_x, valid_x]))
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    train = xgb.DMatrix(train_x, label=train_y)
    valid = xgb.DMatrix(test_x, label=test_y)
    evals = [(train, "train"), (valid, "valid")]

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
    early_stop_rounds = 8

    print "train", mall_id
    params = {
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
    }

    bst = xgb.train(params,
                    train,
                    n_round,
                    evals=evals,
                    early_stopping_rounds=early_stop_rounds)

    p = bst.predict(valid)
    print "lda decomposition xgboost acc", acc(p,test_y)
    # bst = xgb.train(params, n_round=bst.best_iteration )


def main4(mall_id): # SVM
    train_all = load_train()
    shop_info = load_shop_info()
    shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
    num_class = len(shops)

    train = train_all[train_all.mall_id == mall_id]
    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)

    df, train_cache, test_cache = get_wifi_cache(mall_id)
    train_matrix = train_cache[2]
    train_x, test_x, train_y, test_y = train_test_split(train_matrix, y)

    svc = SVC().fit(train_x, train_y)
    p = svc.predict(test_x)
    print "svc acc", acc(p,test_y)
    exit(1)
    scala = 1
    # lda = LinearDiscriminantAnalysis(n_components=num_class).fit(np.concatenate([train_x, valid_x]), np.concatenate([train_y,valid_y]))
    lda = LinearDiscriminantAnalysis(n_components=num_class).fit(train_x, train_y)
    # lda.predict(test_x)
    p = lda.predict(test_x)

    print "LDA", acc(p,test_y)
    # pca = PCA(n_components=int(num_class * scala)).fit(np.concatenate([train_x, valid_x]))
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    train = xgb.DMatrix(train_x, label=train_y)
    valid = xgb.DMatrix(test_x, label=test_y)
    evals = [(train, "train"), (valid, "valid")]

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
    early_stop_rounds = 8

    print "train", mall_id
    params = {
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
    }

    bst = xgb.train(params,
                    train,
                    n_round,
                    evals=evals,
                    early_stopping_rounds=early_stop_rounds)

    p = bst.predict(valid)
    print "lda decomposition xgboost acc", acc(p,test_y)
    # bst = xgb.train(params, n_round=bst.best_iteration )


def main5(mall_id): # t-sne
    train_all = load_train()
    shop_info = load_shop_info()
    shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
    num_class = len(shops)

    train = train_all[train_all.mall_id == mall_id]
    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)

    df, train_cache, test_cache = get_wifi_cache(mall_id)
    train_matrix = train_cache[2]
    train_x, test_x, train_y, test_y = train_test_split(train_matrix, y)

    newx = tsne(train_x, 2, 50, 20.0)
    from matplotlib import pyplot as plt
    plt.scatter(newx[:,0], newx[:,1], 20, train_y)
    plt.savefig("../result/fig/tsne_2d.png")
    # print "tsne acc", acc(p, test_y)
    exit(1)
    scala = 1
    # lda = LinearDiscriminantAnalysis(n_components=num_class).fit(np.concatenate([train_x, valid_x]), np.concatenate([train_y,valid_y]))
    lda = LinearDiscriminantAnalysis(n_components=num_class).fit(train_x, train_y)
    # lda.predict(test_x)
    p = lda.predict(test_x)

    print "LDA", acc(p,test_y)
    # pca = PCA(n_components=int(num_class * scala)).fit(np.concatenate([train_x, valid_x]))
    train_x = lda.transform(train_x)
    test_x = lda.transform(test_x)

    train = xgb.DMatrix(train_x, label=train_y)
    valid = xgb.DMatrix(test_x, label=test_y)
    evals = [(train, "train"), (valid, "valid")]

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
    early_stop_rounds = 8

    print "train", mall_id
    params = {
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
    }

    bst = xgb.train(params,
                    train,
                    n_round,
                    evals=evals,
                    early_stopping_rounds=early_stop_rounds)

    p = bst.predict(valid)
    print "lda decomposition xgboost acc", acc(p,test_y)
    # bst = xgb.train(params, n_round=bst.best_iteration )


def main_tvt(mall_id): # LDA
    train_all = load_train()
    shop_info = load_shop_info()
    shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
    num_class = len(shops)

    train = train_all[train_all.mall_id == mall_id]
    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)

    df, train_cache, test_cache = get_wifi_cache(mall_id)
    train_matrix = train_cache[2]
    train_x, test_x, train_y, test_y = train_test_split(train_matrix, y, test_size=0.1)

    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1)

    scala = 1
    # lda = LinearDiscriminantAnalysis(n_components=num_class).fit(np.concatenate([train_x, valid_x]), np.concatenate([train_y,valid_y]))
    lda = LinearDiscriminantAnalysis(n_components=num_class).fit(train_x, train_y)
    # lda.predict(test_x)

    # pca = PCA(n_components=int(num_class * scala)).fit(np.concatenate([train_x, valid_x]))
    train_x = lda.transform(train_x)
    valid_x = lda.transform(valid_x)
    test_x = lda.transform(test_x)

    train = xgb.DMatrix(train_x, label=train_y)
    valid = xgb.DMatrix(valid_x, label=valid_y)
    test = xgb.DMatrix(test_x, label=test_y)
    evals = [(train, "train"), (valid, "valid")]

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
    early_stop_rounds = 8

    print "train", mall_id
    params = {
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
    }

    bst = xgb.train(params,
                    train,
                    n_round,
                    evals=evals,
                    early_stopping_rounds=early_stop_rounds)

    # bst = xgb.train(params, n_round=bst.best_iteration )
    print bst.eval(test)


if __name__ == '__main__':
    main3(mall_id="m_7800")  # m_6803 m_690
