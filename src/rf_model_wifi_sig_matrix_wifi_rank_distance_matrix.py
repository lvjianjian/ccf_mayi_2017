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
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


def main_leave_one_week(offline, mall_ids=-1):
    model_name = "lightgbm_leave_one_week_wifi_matrix_rank_lonlat_matrix"
    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    if mall_ids == -1:
        mall_ids = shop_info.mall_id.unique()
    offline_predicts = []
    offline_reals = []
    all_rowid = {}
    all_predicts = {}
    kfold = 1
    for _ in range(kfold):
        offline_predicts.append({})
        offline_reals.append({})
    for _index, mall_id in enumerate(mall_ids):
        print "train: ", mall_id, " {}/{}".format(_index, len(mall_ids))
        shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
        num_class = len(shops)
        df, train_cache, test_cache = get_wifi_cache(mall_id)
        train_matrix = train_cache[2]
        test_matrix = test_cache[2]

        # 将wifi 信号加上每个sample的最大wifi信号， 屏蔽个体之间接收wifi信号的差异
        train_matrix = np.tile(-train_matrix.max(axis=1, keepdims=True), (1, train_matrix.shape[1])) + train_matrix
        test_matrix = np.tile(-test_matrix.max(axis=1, keepdims=True), (1, test_matrix.shape[1])) + test_matrix

        # wifi rank info
        train = train_all[train_all.mall_id == mall_id]
        test = test_all[test_all.mall_id == mall_id]
        preprocess_basic_wifi(train)
        preprocess_basic_wifi(test)
        sorted_wifi = get_sorted_wifi([train, test])
        d = rank_sorted_wifi(sorted_wifi)
        other_train_wifi_features = []
        other_test_wifi_features = []
        test_use_wifi_in_wifi_rank, train_use_wifi_in_wifi_rank = use_wifi_in_wifi_rank(test, train, d)
        other_train_wifi_features.append(train_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
        other_test_wifi_features.append(test_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
        # print train_use_wifi_in_wifi_rank
        for _top in range(10):
            test_no_use_wifi_in_wifi_rank, train_no_use_wifi_in_wifi_rank = no_use_wifi_in_wifi_rank(test,
                                                                                                     train,
                                                                                                     d,
                                                                                                     _top)
            other_train_wifi_features.append(train_no_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
            other_test_wifi_features.append(test_no_use_wifi_in_wifi_rank.values.reshape((-1, 1)))

        other_train_wifi_feature = np.concatenate(other_train_wifi_features, axis=1)
        other_test_wifi_feature = np.concatenate(other_test_wifi_features, axis=1)

        scala = 1
        pca = PCA(n_components=int(num_class * scala)).fit(train_matrix)
        train_matrix = pca.transform(train_matrix)
        test_matrix = pca.transform(test_matrix)

        test_index = test_cache[0]
        label_encoder = LabelEncoder().fit(shops)
        y = label_encoder.transform(train.shop_id)

        # distance_matrix
        # 加入经纬度 直接经纬度效果很差
        train_lonlats = train[["longitude", "latitude"]].values
        test_lonlats = test[["longitude", "latitude"]].values
        # 用户经纬度与各个商店的距离矩阵
        d = rank_one(train, "shop_id")
        verctors = []
        for _s, _index in d.items():
            _shop = shop_info[shop_info.shop_id == _s][["shop_longitude", "shop_latitude"]].values
            _shop = np.tile(_shop, (train_lonlats.shape[0], 1))
            verctors.append(haversine(train_lonlats[:, 0],
                                      train_lonlats[:, 1],
                                      _shop[:, 0],
                                      _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        distance_matrix = np.concatenate(verctors, axis=1)

        verctors = []
        for _s, _index in d.items():
            _shop = shop_info[shop_info.shop_id == _s][["shop_longitude", "shop_latitude"]].values
            _shop = np.tile(_shop, (test_lonlats.shape[0], 1))
            verctors.append(haversine(test_lonlats[:, 0],
                                      test_lonlats[:, 1],
                                      _shop[:, 0],
                                      _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        test_dis_matrix = np.concatenate(verctors, axis=1)

        pca_dis = PCA(n_components=int(round(num_class / 5))).fit(distance_matrix)
        distance_matrix = pca_dis.transform(distance_matrix)
        test_dis_matrix = pca_dis.transform(test_dis_matrix)
        train_dis_matrix = distance_matrix

        train_matrix = np.concatenate([train_matrix, train_dis_matrix, other_train_wifi_feature], axis=1)
        test_matrix = np.concatenate([test_matrix, test_dis_matrix, other_test_wifi_feature], axis=1)

        print "num_class", num_class
        # 模型参数
        num_leaves = 35
        learning_rate = 0.04
        feature_fraction = 0.7
        bagging_fraction = 0.8
        bagging_freq = 5
        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': ['multi_logloss', 'multi_error'],
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': 0,
            'num_class': num_class
        }

        n_round = 1000
        early_stop_rounds = 20

        print "start k fold"

        # 将训练数据集划分为最后一周作为验证，前几周作为训练，不用kfold

        print "train", mall_id

        _index = 0
        best_iterations = []
        for _train_index, _valid_index in [get_last_one_week_index(train)]:
            _train_x = train_matrix[_train_index]
            _train_y = y[_train_index]
            _train = lgb.Dataset(_train_x, label=_train_y)

            _valid_x = train_matrix[_valid_index]
            _valid_y = y[_valid_index]
            _valid = lgb.Dataset(_valid_x, label=_valid_y, reference=_train)

            rf = RandomForestClassifier(n_estimators=1000,
                                        n_jobs=-1,
                                        verbose=1)

            predict = rf.predict(_valid_x)
            predict = label_encoder.inverse_transform(predict)
            offline_predicts[_index][mall_id] = predict
            offline_reals[_index][mall_id] = label_encoder.inverse_transform(_valid_y)
            _index += 1
        if not offline:  # 线上
            best_iteration = int(np.mean(best_iterations))
            train = lgb.Dataset(train_matrix, label=y)
            bst = lgb.train(params, train, best_iteration)
            predict = np.argmax(bst.predict(test_matrix, best_iteration), axis=1).astype(int)
            predict = label_encoder.inverse_transform(predict)
            all_predicts[mall_id] = predict
            all_rowid[mall_id] = test_all[np.in1d(test_all.index, test_index)].row_id.values
    result = {}
    for _mall_id in mall_ids:
        accs = []
        for _index in range(kfold):
            _acc = acc(offline_predicts[_index][_mall_id], offline_reals[_index][_mall_id])
            accs.append(_acc)
        print _mall_id + "'s acc is", np.mean(accs)
        result[_mall_id] = np.mean(accs)
    accs = []
    for _index in range(kfold):
        all_predict = np.concatenate(offline_reals[_index].values())
        all_true = np.concatenate(offline_predicts[_index].values())
        _acc = acc(all_predict, all_true)
        accs.append(_acc)
    print "all acc is", np.mean(accs)

    if len(mall_ids) < 97:
        exit(1)

    result["all_acc"] = np.mean(accs)
    path = "../result/offline/{}_f{}_lr{}_leaves{}_ff{}_bf{}_bfq{}_es{}".format(model_name,
                                                                                "num_class_{}".format(scala),
                                                                                learning_rate,
                                                                                num_leaves,
                                                                                feature_fraction,
                                                                                bagging_fraction,
                                                                                bagging_freq,
                                                                                early_stop_rounds)
    save_acc(result, path, None)

    if not offline:
        all_rowid = np.concatenate(all_rowid.values())
        all_predict = np.concatenate(all_predicts.values())
        result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
        result.sort_values(by="row_id", inplace=True)
        path = "../result/online/{}_f{}_lr{}_leaves{}_ff{}_bf{}_bfq{}_es{}".format(model_name,
                                                                                   "num_class_{}".format(scala),
                                                                                   learning_rate,
                                                                                   num_leaves,
                                                                                   feature_fraction,
                                                                                   bagging_fraction,
                                                                                   bagging_freq,
                                                                                   early_stop_rounds)
        save_result(result, path, None)


if __name__ == '__main__':
    # main(offline=False)
    main_leave_one_week(offline=False)  # m_2467 # mall_ids=["m_690", "m_7168", "m_1375", "m_4187", "m_1920", "m_2123"]
