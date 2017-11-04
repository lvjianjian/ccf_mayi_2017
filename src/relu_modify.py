#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-1, 22:56

@Description:

@Update Date: 17-11-1, 22:56
"""

from util import *

def user_modify(_p, _p_t):
    # 在训练集中发现总是在一家商店且同时存在于测试集的用户
    preprocess_basic_time(_p)
    preprocess_basic_time(_p_t)
    count = _p.groupby(["user_id"])["shop_id"].count().reset_index().sort_values(by="shop_id", ascending=False)

    def one_shop(x, p):
        _p = p[p.user_id == x]
        shops = _p.shop_id.unique()
        days = _p.dt.dt.day.unique()
        if len(shops) == 1 and len(days) != 1:
            return x, shops[0]
        else:
            return ""

    count = count[count.shop_id >= 15]
    count = count.user_id.map(lambda x: one_shop(x, _p))
    count = count[count != ""]

    def in_test_and_not_in_one_day(x, test_user_ids):
        if x[0] in test_user_ids:
            return x
        else:
            return ""

    count = count.map(lambda x: in_test_and_not_in_one_day(x, _p_t.user_id.unique()))
    count = count[count != ""]
    return dict(list(count))


def shop_modify(_p):
    # 在训练集中发现总是在一家商店且同时存在于测试集的用户
    _t = _p[_p.dt.dt.hour <= 8]
    if _t.shop_id.unique().shape[0] == 1 and _t.shape[0] > 2:
        return _t.shop_id.unique()[0]
    else:
        return None
