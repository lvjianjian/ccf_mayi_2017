#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-29, 22:22

@Description:

@Update Date: 17-10-29, 22:22
"""

from util import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, fmin, tpe, rand, space_eval
import os, yaml


def main_leave_one_week(offline, mall_ids=-1, use_hyperopt=False, default_scala=2):
    model_name = "rf_leave_one_week_wifi_matrix_rank_lonlat_matrix"
    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    if mall_ids == -1:
        mall_ids = shop_info.mall_id.unique()
    offline_predicts = []
    offline_reals = []
    all_rowid = {}
    all_predicts = {}
    if os.path.exists("../data/best_scala/best_scala_{}.yaml".format(model_name)):
        best_scala = yaml.load(open("../data/best_scala/best_scala_{}.yaml".format(model_name), "r"))
    else:
        best_scala = {}
    kfold = 1
    for _ in range(kfold):
        offline_predicts.append({})
        offline_reals.append({})
    for _index, mall_id in enumerate(mall_ids):
        print "train: ", mall_id, " {}/{}".format(_index, len(mall_ids))
        shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
        num_class = len(shops)
        df, train_cache, test_cache = get_wifi_cache2(mall_id)
        train_matrix = train_cache[2]
        test_matrix = test_cache[2]
        train_matrix_origin = train_matrix.copy()
        test_matrix_origin = test_matrix.copy()
        choose_strong_wifi_index_set = set()

        for _sig_max, _sig_num in zip([-90], [6]):
            strong_sig_index = zip(range(train_matrix.shape[0]),
                                   list((train_matrix > _sig_max).sum(axis=0)))
            strong_sig_index = sorted(strong_sig_index, key=lambda x: -x[1])
            strong_sig_worst = _sig_num
            for _index in range(len(strong_sig_index)):
                if strong_sig_index[_index][1] < strong_sig_worst:
                    break
            strong_sig_choose = _index - 1
            choose_strong_wifi_index = [_wi[0] for _wi in strong_sig_index[:strong_sig_choose]]

            choose_strong_wifi_index_set = choose_strong_wifi_index_set.union(set(choose_strong_wifi_index))
            print len(choose_strong_wifi_index_set)
        # print choose_strong_wifi_index
        choose_strong_wifi_index = list(choose_strong_wifi_index_set)
        train_strong_matrix = train_matrix[:, choose_strong_wifi_index]
        test_strong_matrix = test_matrix[:, choose_strong_wifi_index]

        # 将wifi 信号加上每个sample的最大wifi信号， 屏蔽个体之间接收wifi信号的差异
        train_matrix = np.tile(-train_matrix.max(axis=1, keepdims=True), (1, train_matrix.shape[1])) + train_matrix
        test_matrix = np.tile(-test_matrix.max(axis=1, keepdims=True), (1, test_matrix.shape[1])) + test_matrix

        # wifi rank info
        train = train_all[train_all.mall_id == mall_id]
        test = test_all[test_all.mall_id == mall_id]
        preprocess_basic_wifi(train)
        preprocess_basic_wifi(test)
        other_train_wifi_features = []
        other_test_wifi_features = []
        sorted_wifi_all = get_sorted_wifi_just_train(train, test)

        take = 10
        for _split in [5, 10, 15, 20, 30, 50, 75, 100, 150, 250, 500, 750, 1000]:
            for _index in range(len(sorted_wifi_all), 0, -1):
                if sorted_wifi_all[_index - 1][1] >= _split:
                    break
            sorted_wifi = sorted_wifi_all[:_index]
            d = rank_sorted_wifi(sorted_wifi)

            # use
            test_use_wifi_in_wifi_rank, train_use_wifi_in_wifi_rank = use_wifi_in_wifi_rank2(test, train, d)
            other_train_wifi_features.append(train_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
            other_test_wifi_features.append(test_use_wifi_in_wifi_rank.values.reshape((-1, 1)))

            # no use
            # for _top in range(10):
            #     test_no_use_wifi_in_wifi_rank, train_no_use_wifi_in_wifi_rank = no_use_wifi_in_wifi_rank(test,
            #                                                                                              train,
            #                                                                                              d,
            #                                                                                              _top)
            #     other_train_wifi_features.append(train_no_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
            #     other_test_wifi_features.append(test_no_use_wifi_in_wifi_rank.values.reshape((-1, 1)))

            # all
            for _top in range(take):
                test_all_wifi_in_wifi_rank, train_all_wifi_in_wifi_rank = all_wifi_in_wifi_rank2(test,
                                                                                                 train,
                                                                                                 d,
                                                                                                 _top)
                other_train_wifi_features.append(train_all_wifi_in_wifi_rank.values.reshape((-1, 1)))
                other_test_wifi_features.append(test_all_wifi_in_wifi_rank.values.reshape((-1, 1)))

        other_train_wifi_feature = np.concatenate(other_train_wifi_features, axis=1)
        other_test_wifi_feature = np.concatenate(other_test_wifi_features, axis=1)


        other_train_wifi_feature, other_test_wifi_feature = check_wifi_rank(other_train_wifi_feature,
                                                                            other_test_wifi_feature)
        preprocess_basic_time(train)
        preprocess_basic_time(test)
        train_wh_features = train[["weekday", "hour"]].values
        test_wh_features = test[["weekday", "hour"]].values

        train_w_features = train[["weekday"]].values
        test_w_features = test[["weekday"]].values

        train_h_features = train[["hour"]].values
        test_h_features = test[["hour"]].values

        test_index = test_cache[0]
        label_encoder = LabelEncoder().fit(shops)
        y = label_encoder.transform(train.shop_id)

        n_estimetors = 500
        scala = 2

        print "use scala:", scala
        pca = PCA(n_components=int(num_class * scala)).fit(train_matrix)
        train_matrix = pca.transform(train_matrix)
        test_matrix = pca.transform(test_matrix)

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
            verctors.append(
                    haversine(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        distance_matrix = np.concatenate(verctors, axis=1)

        verctors = []
        for _s, _index in d.items():
            _shop = shop_info[shop_info.shop_id == _s][["shop_longitude", "shop_latitude"]].values
            _shop = np.tile(_shop, (test_lonlats.shape[0], 1))
            verctors.append(
                    haversine(test_lonlats[:, 0], test_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        test_dis_matrix = np.concatenate(verctors, axis=1)

        pca_dis_scala = 3
        pca_dis = PCA(n_components=int(round(num_class / pca_dis_scala))).fit(distance_matrix)
        distance_matrix = pca_dis.transform(distance_matrix)
        test_dis_matrix = pca_dis.transform(test_dis_matrix)
        train_dis_matrix = distance_matrix

        lonlat_pca = PCA().fit(np.concatenate([train_lonlats, test_lonlats]))
        _p_train_pcas = lonlat_pca.transform(train_lonlats)
        _p_test_pcas = lonlat_pca.transform(test_lonlats)

        # train_matrix = other_train_wifi_feature
        # test_matrix = other_test_wifi_feature

        # train_matrix = np.concatenate([train_matrix,
        #                                train_dis_matrix,
        #                                other_train_wifi_feature],
        #                               axis=1)
        # test_matrix = np.concatenate([test_matrix,
        #                               test_dis_matrix,
        #                               other_test_wifi_feature],
        #                              axis=1)

        _train_index, _valid_index = get_last_one_week_index(train)


        train_matrix2 = np.concatenate([train_strong_matrix,
                                       train_lonlats,
                                        # other_train_wifi_feature
                                        # train_w_features,
                                        # train_wh_features,
                                        ], axis=1)

        print "num_class", num_class

        # kfold
        print "train", mall_id

        _index = 0
        for _train_index, _valid_index in [(_train_index, _valid_index)]:
            _train_x = train_matrix2[_train_index]
            _train_y = y[_train_index]

            _valid_x = train_matrix2[_valid_index]
            _valid_y = y[_valid_index]


            rf = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
            rf.fit(_train_x, _train_y)
            predict = rf.predict(_valid_x)
            print mall_id + "'s top 1 acc is", acc(predict, _valid_y)



            exit(1)
            predict = rf.predict(_valid_x)
            print mall_id + "'s top 1 acc is", acc(predict,_valid_y)
            predict = label_encoder.inverse_transform(predict)
            offline_predicts[_index][mall_id] = predict
            offline_reals[_index][mall_id] = label_encoder.inverse_transform(_valid_y)
            _index += 1
            topk = 5
            topk2 = 10
            def get_max_index(predict_proba, k=1):
                predict_proba = predict_proba.copy()
                r = []
                for _ in range(k):
                    index = np.argmax(predict_proba, axis=1)
                    r.append(index)
                    index = zip(range(predict_proba.shape[0]), list(index))
                    for _i1, _i2 in index:
                        predict_proba[_i1, _i2] = 0
                if k == 1:
                    return r
                else:
                    return r

            def get_label_and_proba(predict_proba,_clss):
                maj = np.apply_along_axis(lambda x: [_clss.take(np.argmax(x)),x[np.argmax(x)]],
                                          axis=1,
                                          arr=predict_proba)

                return maj

            predict_proba = rf.predict_proba(_train_x)
            labels_probas = get_label_and_proba(predict_proba, rf.classes_)
            predict = labels_probas[:, 0]
            probas = labels_probas[:, 1]

            threshold = 0.4
            print "all", predict.shape[0]
            print "< {}".format(threshold), (probas < threshold).sum()
            print acc(predict[probas < threshold], _train_y[probas < threshold])

            exit(1)
            predict_list = get_max_index(predict_proba, k=topk)
            predict_list = [rf.classes_[predict] for predict in predict_list]
            print mall_id + "'s top {} acc is".format(topk), topk_acc(predict_list, _valid_y)

            predict_list = get_max_index(predict_proba, k=topk2)
            predict_list = [rf.classes_[predict] for predict in predict_list]



            print mall_id + "'s top {} acc is".format(topk2), topk_acc(predict_list, _valid_y)

            exit(1)

            # 第二次
            _train_x = train_matrix2[_train_index]
            _train_y = y[_train_index]

            _valid_x = train_matrix2[_valid_index]
            _valid_y = y[_valid_index]

            rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1)
            rf.fit(_train_x, _train_y)

            predict_proba = rf.predict_proba(_valid_x)

            def get_label(predict_proba, predict_list,clss):
                # index2cls = zip(range(len(clss)), clss)
                cls2index = dict(zip(clss, range(len(clss))))
                r = []
                for i in range(predict_proba.shape[0]):
                    print i
                    p = predict_proba[i, :]
                    _label = -1
                    _proba = 0
                    for j in range(len(predict_list)):
                        _l = predict_list[j][i]
                        print _l, p[cls2index[_l]]
                        if p[cls2index[_l]] > _proba:
                            _proba = p[cls2index[_l]]
                            _label = _l
                    r.append(_label)
                return np.asarray(r)

            predict = get_label(predict_proba, predict_list,rf.classes_)

            print mall_id + "'s top 1 acc is", acc(predict, _valid_y)



        if not offline:  # 线上
            rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1)
            rf.fit(train_matrix, y)
            predict = rf.predict(test_matrix)
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

    if len(best_scala) != 0:
        scala = "hyperopt"

    if len(mall_ids) < 50:
        exit(1)

    result["all_acc"] = np.mean(accs)
    path = "../result/offline/{}_f{}_es{}".format(model_name,
                                                  "num_class_{}".format(scala),
                                                  n_estimetors)
    save_acc(result, path, None)

    if use_hyperopt:
        yaml.dump(best_scala, open("../data/best_scala/best_scala_{}.yaml".format(model_name), "w"))

    if not offline:
        all_rowid = np.concatenate(all_rowid.values())
        all_predict = np.concatenate(all_predicts.values())
        result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
        result.sort_values(by="row_id", inplace=True)
        path = "../result/online/{}_f{}_es{}".format(model_name,
                                                     "num_class_{}".format(scala),
                                                     n_estimetors)
        save_result(result, path, None)


if __name__ == '__main__':
    # main(offline=False)
    main_leave_one_week(offline=True,
                        mall_ids=["m_6803"],
                        use_hyperopt=False,
                        default_scala=1)  # m_2467 # mall_ids=["m_690", "m_7168", "m_1375", "m_4187", "m_1920", "m_2123"]
