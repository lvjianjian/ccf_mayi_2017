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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesRegressor
from mlxtend.classifier import StackingCVClassifier
from sklearn.multiclass import OneVsRestClassifier, OneVsOneClassifier
from sklearn.linear_model import LogisticRegression

def main_leave_one_week(offline, mall_ids=-1, save_offline_predict=False):
    model_name = "stack_balance_strong_matrix_lonlat_wh"
    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    if mall_ids == -1:
        mall_ids = shop_info.mall_id.unique()
    offline_predicts = {}
    all_rowid = {}
    offline_reals = {}
    all_predicts = {}

    for _index, mall_id in enumerate(mall_ids):
        print "train: ", mall_id, " {}/{}".format(_index + 1, len(mall_ids))
        shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
        train = train_all[train_all.mall_id == mall_id]
        test = test_all[test_all.mall_id == mall_id]

        # y label encoder
        y = train.shop_id.values
        label_encoder = LabelEncoder().fit(y)
        y = label_encoder.transform(y)

        num_class = len(shops)
        print "num_class", num_class

        # all wifi matrix
        df, train_cache, test_cache = get_wifi_cache2(mall_id)
        train_matrix_origin_all = train_cache[2]
        test_matrix_origin_all = test_cache[2]
        test_index = test_cache[0]

        # choose_strong_wifi_index
        strong_wifi_index = choose_strong_wifi_index(-90, 6, train_matrix_origin_all)
        train_strong_matrix = train_matrix_origin_all[:, strong_wifi_index]
        test_strong_matrix = test_matrix_origin_all[:, strong_wifi_index]

        # train valid split and get index
        _train_index, _valid_index = get_last_one_week_index(train)

        # weekday and hour
        preprocess_basic_time(train)
        preprocess_basic_time(test)
        preprocess_basic_wifi(train)
        preprocess_basic_wifi(test)
        train_time_features = train[["weekday", "hour", "is_weekend"]].values
        test_time_features = test[["weekday", "hour", "is_weekend"]].values
        train_wh_features = train[["weekday", "hour"]].values
        test_wh_features = test[["weekday", "hour"]].values

        # 是否连接wifi
        train_connect_wifi = (train.basic_wifi_info.map(lambda x: len(x[1])).values > 0).astype(int).reshape(-1,1)
        test_connect_wifi = (test.basic_wifi_info.map(lambda x: len(x[1])).values > 0).astype(int).reshape(-1,1)


        # 搜到的wifi数量
        train_search_wifi_size = train.basic_wifi_info.map(lambda x: x[0]).values.reshape(-1, 1)
        test_search_wifi_size= test.basic_wifi_info.map(lambda x: x[0]).values.reshape(-1, 1)


        # lon lat
        train_lonlats = train[["longitude", "latitude"]].values
        test_lonlats = test[["longitude", "latitude"]].values

        # concatenate train/test features
        train_matrix = np.concatenate([train_strong_matrix,
                                       train_lonlats,
                                       train_wh_features,
                                       # train_connect_wifi,
                                       # train_search_wifi_size
                                       ],
                                      axis=1)

        test_matrix = np.concatenate([test_strong_matrix,
                                      test_lonlats,
                                      test_wh_features,
                                      # test_connect_wifi,
                                      # test_search_wifi_size
                                      ],
                                     axis=1)

        # train valid get
        _train_x = train_matrix[_train_index]
        _train_y = y[_train_index]
        _valid_x = train_matrix[_valid_index]
        _valid_y = y[_valid_index]

        # stack base model
        def get_model1():
            model1 = RandomForestClassifier(n_estimators=500,
                                            n_jobs=-1,
                                            class_weight="balanced")
            return model1

        def get_model2():
            model2 = OneVsRestClassifier(estimator=RandomForestClassifier(n_estimators=188,
                                                                          n_jobs=-1,
                                                                          class_weight="balanced"))
            return model2

        # stack meta model
        def get_meta_model():
            meta_model = RandomForestClassifier(n_estimators=777,
                                                n_jobs=-1,
                                                class_weight="balanced")
            return meta_model

        # stack cv
        cv = 3

        # offline
        # expansion train
        _x, _y = expansion(_train_x, _train_y, cv)
        stack = StackingCVClassifier([get_model1(), get_model2()],
                                     get_meta_model(),
                                     use_probas=True,
                                     use_features_in_secondary=True,
                                     cv=cv)
        stack.fit(_x, _y)
        best_predict = stack.predict(_valid_x)

        predict = label_encoder.inverse_transform(best_predict)
        offline_predicts[mall_id] = predict
        _real_y = label_encoder.inverse_transform(_valid_y)
        offline_reals[mall_id] = _real_y
        print mall_id + "'s acc is", acc(predict, _real_y)

        # online
        if not offline:
            # expansion train
            _x, _y = expansion(train_matrix, y, cv)
            stack = StackingCVClassifier([get_model1(), get_model2()],
                                         get_meta_model(),
                                         use_probas=True,
                                         use_features_in_secondary=True,
                                         cv=cv)

            stack.fit(_x, _y)
            predict = stack.predict(test_matrix)
            predict = label_encoder.inverse_transform(predict)
            all_predicts[mall_id] = predict
            all_rowid[mall_id] = test_all[np.in1d(test_all.index, test_index)].row_id.values

    # offline acc result
    result = {}
    for _mall_id in mall_ids:
        _acc = acc(offline_predicts[_mall_id], offline_reals[_mall_id])
        print _mall_id + "'s acc is", _acc
        result[_mall_id] = _acc

        if save_offline_predict:
            pd.DataFrame({"predict": offline_predicts[_mall_id],
                          "real": offline_reals[_mall_id]}).to_csv(
                    "../result/offline_predict/{}.csv".format(_mall_id),
                    index=None)

    all_predict = np.concatenate(offline_reals.values())
    all_true = np.concatenate(offline_predicts.values())
    _acc = acc(all_predict, all_true)
    print "all acc is", _acc

    if len(mall_ids) < 50:
        exit(1)

    result["all_acc"] = _acc
    path = "../result/offline/{}".format(model_name)
    save_acc(result, path, None)

    # online save result
    if not offline:
        all_rowid = np.concatenate(all_rowid.values())
        all_predict = np.concatenate(all_predicts.values())
        result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
        result.sort_values(by="row_id", inplace=True)
        path = "../result/online/{}".format(model_name)
        save_result(result, path, None)


if __name__ == '__main__':
    # main(offline=False)
    main_leave_one_week(offline=False,
                        mall_ids=-1,
                        # "m_8093", "m_4572", "m_6803"
                        save_offline_predict=False)  # m_2467 # mall_ids=["m_690", "m_7168", "m_1375", "m_4187", "m_1920", "m_2123"]
