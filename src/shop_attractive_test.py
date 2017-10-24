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


def main(mall_id):
    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    shop_info = shop_info[shop_info.mall_id == mall_id]
    shops = shop_info.shop_id.values
    num_class = len(shops)

    train = train_all[train_all.mall_id == mall_id]
    test = test_all[test_all.mall_id == mall_id]
    label_encoder = LabelEncoder().fit(shops)
    y = label_encoder.transform(train.shop_id)
    user_id = train.user_id.values

    preprocess_basic_time(train)
    preprocess_basic_time(test)
    hour_split = [1, 2, 4, 6]
    name2s = ["weekday"]
    for _h in hour_split:
        train["hour_{}".format(_h)] = (train.hour / _h).astype(int)
        test["hour_{}".format(_h)] = (test.hour / _h).astype(int)
        name2s.append("hour_{}".format(_h))
    train_matrixs = []
    test_matrixs = []
    for _name2 in name2s:
        train_matrix, test_matrix = shop_attractive_matrix("shop_id", _name2, train, test, shops)
        train_matrixs.append(train_matrix)
        test_matrixs.append(test_matrix)
        train_matrix, test_matrix = shop_attractive_matrix("category_id", _name2, train, test, shop_info.category_id.unique())
        train_matrixs.append(train_matrix)
        test_matrixs.append(test_matrix)

    train_matrix = np.concatenate(train_matrixs,axis=1)
    pca = PCA(n_components=num_class/5).fit(train_matrix)
    train_matrix = pca.transform(train_matrix)
    # train_x, test_x, train_y, test_y, train_user_id, test_user_id = train_test_split(train_matrix, y, user_id, test_size=0.1)
    #
    # train_x, valid_x, train_y, valid_y, train_user_id, valid_user_id = train_test_split(train_x, train_y, train_user_id, test_size=0.1)


    train = lgb.Dataset(train_matrix, y)

    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'metric': ['multi_logloss', 'multi_error'],
        'num_leaves': 31,
        'learning_rate': 0.02,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
        'num_class': num_class
    }
    n_round = 200
    early_stop_rounds = 2
    print "train", mall_id
    his = lgb.cv(params,
                 train,
                 n_round,
                 early_stopping_rounds=early_stop_rounds,
                 metrics=["multi_error"])
    print 1 - np.mean(his["multi_error-mean"])



if __name__ == '__main__':
    main("m_690")
