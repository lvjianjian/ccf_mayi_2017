#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-22, 21:44

@Description:

@Update Date: 17-10-22, 21:44
"""

from util import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb


def main_tvt(mall_id):  # train valid test
    train_all = load_train()
    shop_info = load_shop_info()
    shop_info = shop_info[shop_info.mall_id == mall_id]
    shops = shop_info.shop_id.values
    num_class = len(shops)

    train = train_all[train_all.mall_id == mall_id]
    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)
    user_id = train.user_id.values
    # df, train_cache, test_cache = get_wifi_cache(mall_id)

    # 加入经纬度 直接经纬度效果很差
    train_lonlats = train[["longitude", "latitude"]].values

    # 用户经纬度与各个商店的距离矩阵
    d = rank_one(train, "shop_id")
    verctors = []
    for _s, _index in d.items():
        _shop = shop_info[shop_info.shop_id == _s][["shop_longitude", "shop_latitude"]].values
        _shop = np.tile(_shop, (train_lonlats.shape[0], 1))
        verctors.append(haversine(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
    distance_matrix = np.concatenate(verctors, axis=1)

    verctors = []
    for _s, _index in d.items():
        _shop = shop_info[shop_info.shop_id == _s][["shop_longitude", "shop_latitude"]].values
        _shop = np.tile(_shop, (train_lonlats.shape[0], 1))
        verctors.append(haversine(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
    bearing_matrix = np.concatenate(verctors, axis=1)

    df, train_cache, test_cache = get_wifi_cache(mall_id)
    train_matrix = train_cache[2]
    train_matrix = np.tile(-train_matrix.max(axis=1, keepdims=True), (1, train_matrix.shape[1])) + train_matrix
    # print train_matrix.max(axis=1)
    # exit(1)
    train_index = train.index
    # train_index = user_id[np.in1d(train_index, user_id.index)]
    # print train_index.values
    train0_x, test0_x, train_x, test_x, train2_x, test2_x, train_y, test_y, train_index, test_index, train_user_id, test_user_id = train_test_split(
            train_matrix,distance_matrix, bearing_matrix, y, train_index, user_id,
            test_size=0.1)

    train0_x, valid0_x, train_x, valid_x, train2_x, valid2_x, train_y, valid_y, train_index, valid_index, train_user_id, valid_user_id = train_test_split(
            train0_x, train_x, train2_x, train_y, train_index, train_user_id,
            test_size=0.1)

    scala = 1
    # lda = LinearDiscriminantAnalysis(n_components=num_class).fit(np.concatenate([train_x, valid_x]), np.concatenate([train_y,valid_y]))
    # lda = LinearDiscriminantAnalysis(n_components=num_class).fit(train_x, train_y)
    # lda.predict(test_x)

    pca0 = PCA(n_components=int(num_class * scala)).fit(train0_x)
    train0_x = pca0.transform(train0_x)
    valid0_x = pca0.transform(valid0_x)
    test0_x = pca0.transform(test0_x)

    pca = PCA(n_components=int(round(num_class / 5))).fit(train_x)
    train_x = pca.transform(train_x)
    valid_x = pca.transform(valid_x)
    test_x = pca.transform(test_x)

    pca2 = PCA(n_components=int(20)).fit(train2_x)
    train2_x = pca2.transform(train2_x)
    valid2_x = pca2.transform(valid2_x)
    test2_x = pca2.transform(test2_x)

    # train_x = np.concatenate([train_x, train2_x], axis=1)
    # valid_x = np.concatenate([valid_x, valid2_x], axis=1)
    # test_x = np.concatenate([test_x, test2_x], axis=1)

    train_x = np.concatenate([train_x, train0_x], axis=1)
    valid_x = np.concatenate([valid_x, valid0_x], axis=1)
    test_x = np.concatenate([test_x, test0_x], axis=1)

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
    early_stop_rounds = 6

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


    p = bst.predict(test).astype(int)
    print acc(p, test_y)
    exit(1)
    test_y = label_encoder.inverse_transform(test_y)
    p = label_encoder.inverse_transform(p)

    test_user_id = np.concatenate([test_user_id, train_user_id])
    test_y = np.concatenate([test_y, label_encoder.inverse_transform(train_y)])
    p = np.concatenate([p, label_encoder.inverse_transform(bst.predict(train).astype(int))])
    test_index = np.concatenate([test_index, train_index])

    test_user_id = np.concatenate([test_user_id, valid_user_id])
    test_y = np.concatenate([test_y, label_encoder.inverse_transform(valid_y)])
    p = np.concatenate([p, label_encoder.inverse_transform(bst.predict(valid).astype(int))])
    test_index = np.concatenate([test_index, valid_index])

    # print test_index.shape
    # print test_y.shape
    # print p.shape
    r = pd.DataFrame({"real": test_y, "predict": p, "user_id": test_user_id}, index=test_index)
    r.to_csv("../result/{}/test.csv".format(mall_id))


if __name__ == '__main__':
    main_tvt(mall_id="m_615")  # m_6803 m_690
