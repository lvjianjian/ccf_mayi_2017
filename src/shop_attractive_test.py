#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-23, 23:13

@Description:

@Update Date: 17-10-23, 23:13
"""
from util import *
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier

def shop_attractive_matrix(name, name2, train, test, shops):
    max_v = train[name2].max() + 1
    count = train.groupby([name, name2]).count()["user_id"].reset_index()
    d = zip(shops, range(len(shops)))
    d = dict(d)
    matrix = np.zeros((max_v, len(shops)))
    count["label_index"] = count[name].map(lambda x: d[x])
    x = count[[name2, "label_index", "user_id"]].values
    for _x in x:
        matrix[_x[0], _x[1]] = _x[2]
    matrix = pd.DataFrame(matrix)
    matrix[name2] = range(max_v)
    train_matrix = pd.merge(train[[name2]], matrix, how="left", on=name2).drop(name2, axis=1).values
    test_matrix = pd.merge(test[[name2]], matrix, how="left", on=name2).drop(name2, axis=1).values
    return train_matrix, test_matrix


def main(mall_ids = -1):
    train_all = load_train()
    test_all = load_testA()
    shop_info_all = load_shop_info()

    kfold = 1
    offline_predicts = []
    offline_reals = []
    for _ in range(kfold):
        offline_predicts.append({})
        offline_reals.append({})
    for mall_id in mall_ids:
        shop_info = shop_info_all[shop_info_all.mall_id == mall_id]
        shops = shop_info.shop_id.values
        num_class = len(shops)

        train = train_all[train_all.mall_id == mall_id]
        test = test_all[test_all.mall_id == mall_id]
        label_encoder = LabelEncoder().fit(shops)
        y = label_encoder.transform(train.shop_id)
        user_id = train.user_id.values

        preprocess_basic_time(train)
        preprocess_basic_time(test)
        hour_split = [1, 2, 4, 6, 12]
        name2s = ["weekday"]
        for _h in hour_split:
            train["hour_{}".format(_h)] = (train.hour / _h).astype(int)
            test["hour_{}".format(_h)] = (test.hour / _h).astype(int)
            name2s.append("hour_{}".format(_h))
        train_matrixs = []
        test_matrixs = []
        for _name2 in name2s:
            # train_matrix, test_matrix = shop_attractive_matrix("shop_id", _name2, train, test, shops)
            # train_matrixs.append(train_matrix)
            # test_matrixs.append(test_matrix)
            train_matrix, test_matrix = shop_attractive_matrix("category_id", _name2, train, test, shop_info.category_id.unique())
            train_matrixs.append(train_matrix)
            test_matrixs.append(test_matrix)

        train_matrix = np.concatenate(train_matrixs,axis=1)
        pca = PCA(n_components=num_class).fit(train_matrix)
        train_matrix = pca.transform(train_matrix)

        _train_index, _valid_index = get_last_one_week_index(train)

        n_estimetors = 1000

        _index = 0
        for _train_index, _valid_index in [(_train_index, _valid_index)]:
            _train_x = train_matrix[_train_index]
            _train_y = y[_train_index]

            _valid_x = train_matrix[_valid_index]
            _valid_y = y[_valid_index]

            rf = RandomForestClassifier(n_estimators=n_estimetors, n_jobs=-1,max_features=None)
            rf.fit(_train_x, _train_y)

            predict = rf.predict(_valid_x)
            predict = label_encoder.inverse_transform(predict)
            offline_predicts[_index][mall_id] = predict
            offline_reals[_index][mall_id] = label_encoder.inverse_transform(_valid_y)

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

    # train_x, test_x, train_y, test_y, train_user_id, test_user_id = train_test_split(train_matrix, y, user_id, test_size=0.1)
    #
    # train_x, valid_x, train_y, valid_y, train_user_id, valid_user_id = train_test_split(train_x, train_y, train_user_id, test_size=0.1)


    # train = lgb.Dataset(train_matrix, y)
    #
    # params = {
    #     'task': 'train',
    #     'boosting_type': 'gbdt',
    #     'objective': 'multiclass',
    #     'metric': ['multi_logloss', 'multi_error'],
    #     'num_leaves': 31,
    #     'learning_rate': 0.02,
    #     'feature_fraction': 0.8,
    #     'bagging_fraction': 0.8,
    #     'bagging_freq': 5,
    #     'verbose': 0,
    #     'num_class': num_class
    # }
    # n_round = 200
    # early_stop_rounds = 2
    # print "train", mall_id
    # his = lgb.cv(params,
    #              train,
    #              n_round,
    #              early_stopping_rounds=early_stop_rounds,
    #              metrics=["multi_error"])
    # print 1 - np.mean(his["multi_error-mean"])



if __name__ == '__main__':
    main(["m_8093", "m_4572", "m_6803"])
