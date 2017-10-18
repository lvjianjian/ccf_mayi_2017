#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-13, 14:12

@Description:

@Update Date: 17-10-13, 14:12
"""

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
import scipy.sparse as sp
import scipy

shop_info = None
train_info = None
testA_info = None


def acc(pred, y):
    return float((pred == y).sum()) / pred.shape[0]


def save_result(result_csv, path, features):
    result_csv.to_csv(path + ".csv", index=False)
    with open(path + ".feature", "w") as f:
        f.write(",".join(features))
        f.flush()


def save_acc(result, path, features):
    with open(path + ".feature", "w") as f:
        f.write(",".join(features) + "\n")
        for _mallid in result:
            f.write(str(_mallid) + ":" + str(result[_mallid]) + "\n")
        f.flush()


def load_shop_info():
    global shop_info
    if shop_info is None:
        shop_info = pd.read_csv("../data/训练数据-ccf_first_round_shop_info.csv")
        shop_info.columns = ['shop_id', 'category_id', 'shop_longitude', 'shop_latitude', 'price', 'mall_id']
    return shop_info


def load_train():
    global train_info
    if train_info is None:
        train_info = pd.read_csv("../data/训练数据-ccf_first_round_user_shop_behavior.csv")
        shop_info = load_shop_info()
        train_info = pd.merge(train_info, shop_info, on="shop_id", how="left")
    return train_info


def load_testA():
    global testA_info
    if testA_info is None:
        testA_info = pd.read_csv("../data/AB榜测试集-evaluation_public.csv")
    return testA_info


def preprocess_basic_wifi(data):
    # 处理wifi info
    def wifi_info_preprocess(x):
        all_wifis = x.split(";")
        wifi_size = len(all_wifis)
        no_use_wifis = []
        use_wifis = []
        for _wifi in all_wifis:
            _infos = _wifi.split("|")
            _wifi_name = _infos[0]
            _wifi_signal = int(_infos[1])
            _wifi_use = _infos[2]
            if _wifi_use == "true":
                use_wifis.append((_wifi_name, _wifi_signal))
            elif _wifi_use == "false":
                no_use_wifis.append((_wifi_name, _wifi_signal))

        return (wifi_size, use_wifis, no_use_wifis)

    wifi_infos = data.wifi_infos.map(lambda x: wifi_info_preprocess(x))
    data.loc[:, "basic_wifi_info"] = wifi_infos
    # 搜索到的wifi数量
    data.loc[:, "wifi_size"] = wifi_infos.map(lambda x: x[0])
    # 使用的wifi数量
    data.loc[:, "use_wifi_size"] = wifi_infos.map(lambda x: len(x[1]))
    # 没有使用的wifi数量
    data.loc[:, "no_use_wifi_size"] = wifi_infos.map(lambda x: len(x[2]))
    # 使用的wifi数量占全部搜索到的比例
    data.loc[:, "use_wifi_freq"] = data.use_wifi_size / data.wifi_size
    # 没有使用的占全部的比例
    data.loc[:, "no_use_wifi_freq"] = data.no_use_wifi_size / data.wifi_size

    return data


def preprocess_time(data):
    data.loc[:, "dt"] = pd.to_datetime(data.time_stamp)
    return data


def preprocess_lonlat(train, test):
    shop_info = load_shop_info()
    mall_ids = shop_info.mall_id.unique()

    for _mall_id in mall_ids:
        _part_shop = shop_info[shop_info.mall_id == _mall_id]
        _part_train = train[train.mall_id == _mall_id]
        _part_test = test[test.mall_id == _mall_id]

        # 是否在shop附近
        _e = 1 * 10e-6
        lon_max = _part_shop.shop_longitude.max() + _e
        lon_min = _part_shop.shop_longitude.min() - _e
        lat_max = _part_shop.shop_latitude.max() + _e
        lat_min = _part_shop.shop_latitude.min() - _e
        reasonable_part_train = _part_train[(_part_train.longitude > lon_min) &
                                            (_part_train.longitude < lon_max) &
                                            (_part_train.latitude > lat_min) &
                                            (_part_train.latitude < lat_max)]

        train.loc[reasonable_part_train.index, "lon_lat_in_shop"] = 1
        train.loc[_part_train[~np.in1d(_part_train.index, reasonable_part_train.index)].index, "lon_lat_in_shop"] = 0

        reasonable_part_test = _part_test[(_part_test.longitude > lon_min) &
                                          (_part_test.longitude < lon_max) &
                                          (_part_test.latitude > lat_min) &
                                          (_part_test.latitude < lat_max)]

        test.loc[reasonable_part_test.index, "lon_lat_in_shop"] = 1
        test.loc[_part_test[~np.in1d(_part_test.index, reasonable_part_test.index)].index, "lon_lat_in_shop"] = 0

        # 对经纬度进行PCA变换
        pca = PCA().fit(
                np.concatenate([_part_train[["longitude", "latitude"]].values, _part_test[["longitude", "latitude"]]]))
        _p_train_pcas = pca.transform(_part_train[["longitude", "latitude"]].values)
        _p_test_pcas = pca.transform(_part_test[["longitude", "latitude"]].values)
        train.loc[_part_train.index, "pca_lon"] = _p_train_pcas[:, 0]
        train.loc[_part_train.index, "pca_lat"] = _p_train_pcas[:, 1]
        test.loc[_part_test.index, "pca_lon"] = _p_test_pcas[:, 0]
        test.loc[_part_test.index, "pca_lat"] = _p_test_pcas[:, 1]

        # 将shop_info中的商店经纬度聚类
        cluster = MiniBatchKMeans(n_clusters=15, max_iter=500).fit(
                _part_shop[["shop_longitude", "shop_latitude"]].values)
        train.loc[_part_train.index, "cluster_label"] = cluster.predict(_part_train[["longitude", "latitude"]].values)
        test.loc[_part_test.index, "cluster_label"] = cluster.predict(_part_test[["longitude", "latitude"]].values)

    return train, test


def get_all_wifi(x, all_wifis):
    for _x in x[1]:
        all_wifis.append(_x[0])
    for _x in x[2]:
        all_wifis.append(_x[0])


def get_sorted_wifi(datas):
    all_wifis = []
    for _data in datas:
        _data.basic_wifi_info.map(lambda x: get_all_wifi(x, all_wifis))
    c = Counter(all_wifis)
    sorted_wifi = sorted(c.items(), key=lambda x: -x[1])
    return sorted_wifi


def rank_sorted_wifi(sorted_wifi):
    assert isinstance(sorted_wifi, list)
    d = {}  # 将wifi按rank放入dict, 排名从0开始
    for i, w in enumerate(sorted_wifi):
        d[w[0]] = i
    return d


def preprocess_wifi(train, test):
    shop_info = load_shop_info()
    mall_ids = shop_info.mall_id.unique()
    for _mall_id in mall_ids:
        # 对wifi进行rank,选出前shop_size * 4的wifi进行排序
        shop_size = shop_info[shop_info.mall_id == _mall_id].shape[0]
        _part_train = train[train.mall_id == _mall_id]
        _part_test = test[test.mall_id == _mall_id]

        sorted_wifi = get_sorted_wifi([_part_train, _part_test])

        sorted_wifi = sorted_wifi[:shop_size * 4]

        d = rank_sorted_wifi(sorted_wifi)

        # 使用的wifi在wifi_rank中的排名,若有多个使用wifi，选sig强的， 若没有使用wifi，则设置为rank_size
        def use_wifi_rank(x, d):
            size = len(d)
            use_wifi = sorted(x[1], key=lambda x: -x[1])
            if len(use_wifi) >= 1:
                if use_wifi[0][0] in d:
                    return d[use_wifi[0][0]]
                else:
                    return size
            else:
                return size

        use_wifi_in_wifi_rank = _part_train.basic_wifi_info.map(lambda x: use_wifi_rank(x, d))
        train.loc[use_wifi_in_wifi_rank.index, "use_wifi_in_wifi_rank"] = use_wifi_in_wifi_rank
        use_wifi_in_wifi_rank = _part_test.basic_wifi_info.map(lambda x: use_wifi_rank(x, d))
        test.loc[use_wifi_in_wifi_rank.index, "use_wifi_in_wifi_rank"] = use_wifi_in_wifi_rank

        # 未使用的wifi在wifi_rank中的排名, top为信号排名中的第top个,取值0-9
        def no_use_wifi_rank(x, d, top):
            size = len(d)
            use_wifi = sorted(x[2], key=lambda x: -x[1])
            if len(use_wifi) > top:
                if use_wifi[top][0] in d:
                    return d[use_wifi[top][0]]
                else:
                    return size
            else:
                return size

        for _top in range(10):
            no_use_wifi_in_wifi_rank = _part_train.basic_wifi_info.map(lambda x: no_use_wifi_rank(x, d, _top))
            train.loc[no_use_wifi_in_wifi_rank.index, "no_use_wifi_top{}_in_wifi_rank".format(
                    _top)] = no_use_wifi_in_wifi_rank
            no_use_wifi_in_wifi_rank = _part_test.basic_wifi_info.map(lambda x: no_use_wifi_rank(x, d, _top))
            test.loc[no_use_wifi_in_wifi_rank.index, "no_use_wifi_top{}_in_wifi_rank".format(
                    _top)] = no_use_wifi_in_wifi_rank

        # 取wifi排名前100 ,将用户的wifi强度投射进去，不存在为-115
        choose_num = 100

        def wifi_signal_in_top(x, wifi_name):
            worst_sig = -115
            for _x in x[1]:
                if _x[0] == wifi_name:
                    return _x[1]
            for _x in x[2]:
                if _x[0] == wifi_name:
                    return _x[1]
            return worst_sig

        for _i in range(choose_num):
            train.loc[_part_train.index, "top_{}_wifi_sig".format(_i)] = _part_train.basic_wifi_info.map(
                    lambda x: wifi_signal_in_top(x, sorted_wifi[_i][0]))
            test.loc[_part_test.index, "top_{}_wifi_sig".format(_i)] = _part_test.basic_wifi_info.map(
                    lambda x: wifi_signal_in_top(x, sorted_wifi[_i][0]))

        # 对同一个cluster的构建wifi rank
        cluster_wifi_choose_num = 20
        cluster_labels = np.union1d(_part_train.cluster_label.unique(), _part_test.cluster_label.unique())
        for _label in cluster_labels:
            _part_train_cluster = _part_train[_part_train.cluster_label == _label]
            _part_test_cluster = _part_test[_part_test.cluster_label == _label]
            all_wifi_cluster = []
            _part_train_cluster.basic_wifi_info.map(lambda x: get_all_wifi(x, all_wifi_cluster))
            _part_test_cluster.basic_wifi_info.map(lambda x: get_all_wifi(x, all_wifi_cluster))
            c3 = Counter(all_wifi_cluster)
            sorted_wifi_cluster = sorted(c3.items(), key=lambda x: -x[1])
            sorted_wifi_cluster = sorted_wifi_cluster[:cluster_wifi_choose_num]
            d = {}  # 将wifi按rank放入dict, 排名从0开始
            for i, w in enumerate(sorted_wifi_cluster):
                d[w[0]] = i
            use_wifi_in_wifi_rank = _part_train_cluster.basic_wifi_info.map(lambda x: use_wifi_rank(x, d))
            train.loc[use_wifi_in_wifi_rank.index, "cluster_use_wifi_in_wifi_rank"] = use_wifi_in_wifi_rank
            use_wifi_in_wifi_rank = _part_test_cluster.basic_wifi_info.map(lambda x: use_wifi_rank(x, d))
            test.loc[use_wifi_in_wifi_rank.index, "cluster_use_wifi_in_wifi_rank"] = use_wifi_in_wifi_rank

            for _top in range(3):
                no_use_wifi_in_wifi_rank = _part_train_cluster.basic_wifi_info.map(
                        lambda x: no_use_wifi_rank(x, d, _top))
                train.loc[no_use_wifi_in_wifi_rank.index, "cluster_no_use_wifi_top{}_in_wifi_rank".format(
                        _top)] = no_use_wifi_in_wifi_rank
                no_use_wifi_in_wifi_rank = _part_test_cluster.basic_wifi_info.map(
                        lambda x: no_use_wifi_rank(x, d, _top))
                test.loc[no_use_wifi_in_wifi_rank.index, "cluster_no_use_wifi_top{}_in_wifi_rank".format(
                        _top)] = no_use_wifi_in_wifi_rank

    train = train.drop("basic_wifi_info", axis=1)
    test = test.drop("basic_wifi_info", axis=1)
    return train, test


def preprocess():
    train = load_train()
    test = load_testA()
    train = preprocess_basic_wifi(train)
    test = preprocess_basic_wifi(test)

    train = preprocess_time(train)
    test = preprocess_time(test)

    train, test = preprocess_lonlat(train, test)
    train, test = preprocess_wifi(train, test)

    return train, test


def train_split(train, ratio=(8, 1, 1)):
    assert len(ratio) == 3
    sort_train = train.sort_values(by="dt")
    chunk_all = ratio[0] + ratio[1] + ratio[2]
    train_size = float(ratio[0]) / chunk_all
    valid_size = float(ratio[1]) / chunk_all
    sample_all = train.shape[0]
    index1 = int(np.ceil(sample_all * train_size))
    index2 = index1 + int(np.ceil(sample_all * valid_size))
    train = sort_train.iloc[:index1]
    valid = sort_train.iloc[index1:index2]
    test = sort_train.iloc[index2:]
    return train, valid, test


def basic_wifi_map2matrix(x, wifi_matrix, wifi_rank_dict, use_wifis_str):
    index = x[1]
    basic_wifi = x[0]
    use_str = ""
    for x_use in basic_wifi[1]:
        wifi_matrix[index, wifi_rank_dict[x_use[0]]] = x_use[1]
        use_str += (str(wifi_rank_dict[x_use[0]]) + "|")
    for x_no_use in basic_wifi[2]:
        wifi_matrix[index, wifi_rank_dict[x_no_use[0]]] = x_no_use[1]
    if use_str != "":
        use_str = use_str[:-1]
    use_wifis_str.append(use_str)


def wifi_info2csv(datas, names):
    assert isinstance(datas, list)
    assert isinstance(names, list)
    shop_info = load_shop_info()
    mall_ids = shop_info.mall_id.unique()
    for _data in datas:
        if "basic_wifi_info" not in _data.columns:
            preprocess_basic_wifi(_data)
    for _mall_id in mall_ids:
        print _mall_id
        part_datas = [_data[_data.mall_id == _mall_id] for _data in datas]
        sorted_wifi = get_sorted_wifi(part_datas)
        df = pd.DataFrame({"wifi_name": [wifi[0] for wifi in sorted_wifi], "wifi_num": [wifi[1] for wifi in sorted_wifi]})
        df.index.name = "wifi_rank"
        df.to_csv("../data/wifi_info_cache/{}_rank.csv".format(_mall_id))
        d = rank_sorted_wifi(sorted_wifi)
        for _part_data, name in zip(part_datas, names):
            wifi_matrix = np.zeros((_part_data.shape[0], len(sorted_wifi)))
            # wifi_matrix[:] = -115
            use_wifi_str = []
            _part_data.loc[:, "i_index"] = range(_part_data.shape[0])
            _part_data[["basic_wifi_info", "i_index"]].apply(lambda x: basic_wifi_map2matrix(x, wifi_matrix, d, use_wifi_str), axis=1)
            a = np.asarray(use_wifi_str)
            # 用csv 存读取很慢, 将index, usewifi 和matrix 分开存
            np.save("../data/wifi_info_cache/{}_{}_index".format(name, _mall_id), _part_data.index)
            np.save("../data/wifi_info_cache/{}_{}_use_wifi".format(name, _mall_id), a)
            # 用稀疏矩阵存取

            x = sp.csc_matrix(wifi_matrix)
            scipy.save("../data/wifi_info_cache/{}_{}_matrix".format(name, _mall_id), x)


def do_wifi_cache():
    wifi_info2csv([load_train(), load_testA()], ["train", "test"])


if __name__ == '__main__':
    # save_result(None, "../result/test", ["a", "b"])
    do_wifi_cache()
