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
from sklearn.ensemble import RandomForestClassifier
from hyperopt import hp, fmin, tpe, rand, space_eval
import os, yaml
from sklearn.svm import SVC
from mlxtend.classifier import EnsembleVoteClassifier, StackingCVClassifier, StackingClassifier

def main_leave_one_week(offline, mall_ids=-1,
                        use_hyperopt=False,
                        default_scala=2,
                        save_offline_predict=False,
                        choose_input=None,
                        part_shops=None):
    model_name = "diff_input_rf_leave_one_week_wifi_matrix_strong_wifi_matrix_rank2_lonlat_matrix"
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
        train_matrix_origin_all = train_matrix.copy()
        test_matrix_origin_all = test_matrix.copy()

        # print choose_strong_wifi_index
        choose_strong_wifi_index = choose_string_wifi_index(-90, 6, train_matrix)
        train_strong_matrix = train_matrix[:, choose_strong_wifi_index]
        test_strong_matrix = test_matrix[:, choose_strong_wifi_index]

        # wifi rank info
        train = train_all[train_all.mall_id == mall_id]
        test = test_all[test_all.mall_id == mall_id]

        print "shops", len(shops)
        num_class = len(train.shop_id.unique())
        print "train shops", num_class

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

        # lonlat_pca = PCA().fit(np.concatenate([train_lonlats, test_lonlats]))
        # _p_train_pcas = lonlat_pca.transform(train_lonlats)
        # _p_test_pcas = lonlat_pca.transform(test_lonlats)


        n_estimetors = 1000
        max_features = "auto"
        _train_index, _valid_index = get_last_one_week_index(train, part_shops=part_shops)
        print _train_index.shape
        print _valid_index.shape
        argsDict = {}
        if use_hyperopt:
            def objective(argsDict):
                _train_matrix = train_matrix.copy()
                _scala = argsDict["scala"]
                pca = PCA(n_components=int(num_class * _scala)).fit(_train_matrix)
                _train_matrix = pca.transform(_train_matrix)

                _train_matrix = np.concatenate([_train_matrix,
                                                train_dis_matrix,
                                                other_train_wifi_feature],
                                               axis=1)
                _train_x = _train_matrix[_train_index]
                _train_y = y[_train_index]

                _valid_x = _train_matrix[_valid_index]
                _valid_y = y[_valid_index]

                rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1, max_features=max_features)
                rf.fit(_train_x, _train_y)
                y_predict = rf.predict(_valid_x)
                return -acc(y_predict, _valid_y)

            space = {
                "scala": hp.uniform("scala", 0.3, 8)
            }

            best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=12)
            argsDict = space_eval(space, best_sln)
            best_scala[mall_id] = argsDict["scala"]
        else:
            if len(best_scala) == 0:
                argsDict["scala"] = default_scala
            else:
                argsDict["scala"] = best_scala[mall_id]

        scala = argsDict["scala"]
        print "use scala:", scala
        pca = PCA(n_components=int(num_class * scala)).fit(train_matrix)
        train_matrix = pca.transform(train_matrix)
        test_matrix = pca.transform(test_matrix)

        # 时间
        preprocess_basic_time(train)
        preprocess_basic_time(test)
        train_time_features = train[["weekday", "hour"]].values
        test_time_features = test[["weekday", "hour"]].values
        train_h_features = train[["hour"]].values
        test_h_features = test[["hour"]].values
        train_w_features = train[["weekday"]].values
        test_w_features = test[["weekday"]].values
        # train_matrix = np.concatenate([train_matrix,
        #                                train_dis_matrix,
        #                                train_strong_matrix,
        #                                other_train_wifi_feature,
        #                                train_time_features,
        #                                train_lonlats
        #                                ],
        #                               axis=1)
        # test_matrix = np.concatenate([test_matrix,
        #                               test_dis_matrix,
        #                               test_strong_matrix,
        #                               other_test_wifi_feature,
        #                               test_time_features,
        #                               test_lonlats],
        #                              axis=1)

        print "num_class", num_class

        print "train", mall_id

        _index = 0
        for _train_index, _valid_index in [(_train_index, _valid_index)]:

            input1 = [train_matrix_origin_all,
                      train_strong_matrix,
                      train_lonlats,
                      train_time_features,
                      train_matrix,
                      other_train_wifi_feature,
                      train_h_features,
                      train_w_features]

            input1_test = [test_matrix_origin_all,
                           test_strong_matrix,
                           test_lonlats,
                           test_time_features,
                           test_matrix,
                           other_test_wifi_feature,
                           test_h_features,
                           test_w_features]

            if choose_input is None:
                base = []
                _train_y = y[_train_index]
                _valid_y = y[_valid_index]
                best_predict = None
                best_acc = 0
                for _i in range(2):  # 选择用origin_all 还是用 strong_matrix
                    _train_matrix = input1[_i]
                    _train_x = _train_matrix[_train_index]
                    _valid_x = _train_matrix[_valid_index]

                    rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1, random_state=2017)
                    rf.fit(_train_x, _train_y)

                    predict = rf.predict(_valid_x)
                    base.append(acc(predict, _valid_y))
                    if base[_i] > best_acc:
                        best_acc = base[_i]
                        best_predict = predict

                assert len(base) == 2
                if base[0] > base[1]:
                    base_matrix = input1[0]
                    base_test_matrix = input1_test[0]
                    base_acc = base[0]
                    choose = [0]
                else:
                    base_matrix = input1[1]
                    base_test_matrix = input1_test[1]
                    base_acc = base[1]
                    choose = [1]

                # 尝试train_matrix + other_train_wifi_feature
                _train_matrix = np.concatenate([input1[4], input1[5]], axis=1)
                _train_x = _train_matrix[_train_index]
                _valid_x = _train_matrix[_valid_index]

                rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1, random_state=2017)
                rf.fit(_train_x, _train_y)

                predict = rf.predict(_valid_x)
                _acc = acc(predict, _valid_y)
                if _acc > best_acc:
                    best_acc = _acc
                    base_acc = [_acc]
                    best_predict = predict
                    choose = [4, 5]
                    base_matrix = np.concatenate([input1[4], input1[5]], axis=1)
                    base_test_matrix = np.concatenate([input1_test[4], input1_test[5]], axis=1)

                # 再选择是否加入lonlat 和 time
                extra_accs = [base_acc]
                extras = [[2], [3], [2, 3]]
                for extra_info_index in extras:
                    extra_infos = [base_matrix]
                    for _i in extra_info_index:
                        extra_infos.append(input1[_i])
                    _train_matrix = np.concatenate(extra_infos, axis=1)
                    _train_x = _train_matrix[_train_index]

                    _valid_x = _train_matrix[_valid_index]

                    rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1, random_state=2017)
                    rf.fit(_train_x, _train_y)

                    predict = rf.predict(_valid_x)
                    _acc = acc(predict, _valid_y)
                    extra_accs.append(_acc)
                    if _acc > best_acc:
                        best_acc = _acc
                        best_predict = predict
            else:
                t = []
                for _i in choose_input:
                    t.append(input1[_i])

                _train_matrix = np.concatenate(t, axis=1)

                _train_x = _train_matrix[_train_index]
                _valid_x = _train_matrix[_valid_index]
                _train_y = y[_train_index]
                _valid_y = y[_valid_index]

                # from sklearn.preprocessing import MinMaxScaler
                # mms = MinMaxScaler().fit(_train_x)
                # _train_x = mms.transform(_train_x)
                # _valid_x = mms.transform(_valid_x)
                rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1, random_state=2017, max_features=0.7)

                # _train_x, _train_y = expansion(_train_x, _train_y)
                rf.fit(_train_x, _train_y)
                best_predict = rf.predict(_valid_x)

            predict = label_encoder.inverse_transform(best_predict)
            offline_predicts[_index][mall_id] = predict
            _real_y = label_encoder.inverse_transform(_valid_y)
            offline_reals[_index][mall_id] = _real_y
            _index += 1
            print mall_id + "'s acc is", acc(predict, _real_y)

        if not offline:  # 线上
            assert len(extra_accs) == 4
            ch = np.argmax(extra_accs)
            inputs_train = [base_matrix]
            inputs_test = [base_test_matrix]
            if ch > 0:
                for _i in extras[ch - 1]:
                    choose.append(_i)
                    inputs_train.append(input1[_i])
                    inputs_test.append(input1_test[_i])
            print "choose", choose
            train_matrix = np.concatenate(inputs_train, axis=1)
            test_matrix = np.concatenate(inputs_test, axis=1)

            rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1, random_state=2017)
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

        if save_offline_predict:
            pd.DataFrame({"predict": offline_predicts[_index][_mall_id],
                          "real": offline_reals[_index][_mall_id]}).to_csv(
                    "../result/offline_predict/{}_{}.csv".format(_mall_id,
                                                                 _index),
                    index=None)
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
    path = "../result/offline/{}_{}_{}_es{}".format(model_name,
                                                    "pca_scala_{}".format(scala),
                                                    "pca_dis_scala_{}".format(pca_dis_scala),
                                                    n_estimetors)
    save_acc(result, path, None)

    if use_hyperopt:
        yaml.dump(best_scala, open("../data/best_scala/best_scala_{}.yaml".format(model_name), "w"))

    if not offline:
        all_rowid = np.concatenate(all_rowid.values())
        all_predict = np.concatenate(all_predicts.values())
        result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
        result.sort_values(by="row_id", inplace=True)
        path = "../result/online/{}_{}_{}_es{}".format(model_name,
                                                       "num_class_{}".format(scala),
                                                       "pca_dis_scala_{}".format(pca_dis_scala),
                                                       n_estimetors)
        save_result(result, path, None)


if __name__ == '__main__':
    # main(offline=False)
    main_leave_one_week(offline=True,
                        mall_ids=["m_7168"],
                        choose_input=[1, 2, 3],
                        part_shops=None,  # ["s_293704", "s_2310712"]
                        # "m_8093", "m_4572", "m_6803"
                        use_hyperopt=False,
                        default_scala=2,
                        save_offline_predict=True)  # m_2467 # mall_ids=["m_690", "m_7168", "m_1375", "m_4187", "m_1920", "m_2123"]
