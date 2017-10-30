#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-29, 22:05

@Description:

@Update Date: 17-10-29, 22:05
"""

from util import *


def main(mall_id, offline):
    train = load_train()
    test = load_testA()
    shop = load_shop_info()
    train = train[train.mall_id == mall_id]
    test = test[test.mall_id == mall_id]
    shop = shop[shop.mall_id == mall_id]
    df, (train_index, train_use_wifi, train_matrix), \
    (test_index, test_use_wifi, test_matrix) = get_wifi_cache(mall_id=mall_id)

    if offline:
        _train_index, _valid_index = get_last_one_week_index(train)
        test_matrix = train_matrix[_valid_index, :]
        train_matrix = train_matrix[_train_index, :]
        test = train.iloc[_valid_index]
        train = train.iloc[_train_index]
        train_index = train.index

    sig_one_wifi_shop_inverted_index = {}
    train_day = train
    print train_day.shop_id.unique().shape
    print
    day_shop_ids = train_day.shop_id.unique()
    for _s in day_shop_ids:
        train_day_shop_index = train_day[train_day.shop_id == _s].index
        train_day_shop_matrix = train_matrix[np.in1d(train_index, train_day_shop_index)]
        wifi_index = np.argmax(train_day_shop_matrix, axis=1)
        #     print wifi_index
        size = wifi_index.shape[0]
        c = Counter(wifi_index)
        for _wifi_index, _size in c.items():
            if _wifi_index in sig_one_wifi_shop_inverted_index:
                sig_one_wifi_shop_inverted_index[_wifi_index].append((_s, float(_size) / size, _size, size))
            else:
                sig_one_wifi_shop_inverted_index[_wifi_index] = [(_s, float(_size) / size, _size, size)]

    test_sig_max = np.argmax(test_matrix, axis=1)
    test_predict = []
    for _index in test_sig_max:
        if _index in sig_one_wifi_shop_inverted_index.keys():
            #         if len(sig_one_wifi_shop_inverted_index[_index]) == 1:
            #             ensure += 1
            may_shops = sig_one_wifi_shop_inverted_index[_index]
            if len(may_shops) == 1:
                test_predict.append(may_shops[0][0])
            else:
                _predict_shop = ""
                _rate = 0
                for _s, _r, _size, _all_ize in may_shops:
                    if _all_ize != 1 and _r > _rate:
                        _predict_shop = _s
                        _rate = _r

                test_predict.append(_predict_shop)

        else:
            test_predict.append("")

    print acc(np.array(test_predict), test.shop_id.values)


if __name__ == '__main__':
    main("m_8093", offline=True)
