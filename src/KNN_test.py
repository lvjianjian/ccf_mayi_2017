#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-18, 23:08

@Description:

@Update Date: 17-10-18, 23:08
"""
from util import *
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


def main(offline=False, mall_id="m_690"):  # 每个mall 训练一个分类器
    model_name = "KNN"

    train, test = preprocess(mall_id)

    # 划分验证 测试集
    if offline:
        train, valid, test = train_split(train)
    else:
        train, valid, _ = train_split(train, ratio=(9, 1, 0))

    # for _index, _mall_id in enumerate(mall_ids):
    _mall_id = mall_id
    print "train: ", _mall_id
    _train = train
    _valid = valid
    all_shop_id = np.union1d(_train.shop_id.unique(), _valid.shop_id.unique())
    label_encoder = LabelEncoder().fit(all_shop_id)

    features = ["top_{}_wifi_sig".format(_i) for _i in range(top_wifi_sig_num)]
    _train_x = _train[features].values
    _train_y = label_encoder.transform(_train["shop_id"].values)
    _valid_x = _valid[features].values
    _valid_y = label_encoder.transform(_valid["shop_id"].values)
    # 训练
    model = KNeighborsClassifier(n_neighbors=20).fit(_train_x, _train_y)

    # 进行预测
    predicts = {}
    row_ids_or_true = {}
    # for _mall_id in mall_ids:
    # _test = test[test.mall_id == _mall_id]
    # _test_x = _test[features].values
    # _test_DMatrix = xgb.DMatrix(_test_x)
    predict = model.predict(_valid_x)
    # predict = label_encoder.inverse_transform(predict)
    if offline:
        print acc(predict, _valid_y)
    else:
        pass


if __name__ == '__main__':
    main(offline=True, mall_id="m_690")
    # main(offline=False)
