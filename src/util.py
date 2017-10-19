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

# 不同的mall 需要不同的参数 ！！！ 略尴尬，这里怎么选呢？ 改变xgboost subsample 和colsample可以缓解
global_top_wifi_sig_50 = 50  # 在选取前50不降纬作为特征
global_top_wifi_sig_num = "shop_size_4"  # 选取的top wifi sig是shop size的4倍
pca_component_top_wifi_sig = 15  # 对top_wifi_sig进行 pca降纬




def acc(pred, y):
    return float((pred == y).sum()) / pred.shape[0]


def save_result(result_csv, path, features):
    result_csv.to_csv(path + ".csv", index=False)
    if features is not None:
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


def preprocess_basic_time(data):
    data.loc[:, "dt"] = pd.to_datetime(data.time_stamp)
    # 是否会引入序的先验？
    data.loc[:, "weekday"] = data.dt.dt.weekday
    data.loc[:, "hour"] = data.dt.dt.hour

    data.loc[:, "is_weekend"] = np.where((data.weekday == 5) | (data.weekday == 6), 1, 0)
    return data


def preprocess_lonlat(train, test):
    print "preprocess_lonlat"
    mall_ids = train.mall_id.unique()

    for _mall_id in mall_ids:
        print "preprocess_lonlat", _mall_id
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


def wifi_signal_in_top(x, wifi_name):
    worst_sig = -115
    for _x in x[1]:
        if _x[0] == wifi_name:
            return _x[1]
    for _x in x[2]:
        if _x[0] == wifi_name:
            return _x[1]
    return worst_sig


def preprocess_wifi(train, test):
    print "preprocess_wifi"
    mall_ids = train.mall_id.unique()

    train_wifi_matrix_dfs = []
    test_wifi_matrix_dfs = []
    train_wifi_matrix_dfs2 = []
    test_wifi_matrix_dfs2 = []

    for _mall_id in mall_ids:
        print "preprocess_wifi at", _mall_id
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
        df, train_wifi_cache, test_wifi_cache = get_wifi_cache(_mall_id)
        train_matrix = train_wifi_cache[2]
        test_matrix = test_wifi_cache[2]
        top_wifi_sig_num = global_top_wifi_sig_num
        if isinstance(top_wifi_sig_num, str):
            scala = float(top_wifi_sig_num.split("_")[2])
            top_wifi_sig_num = int(shop_size * scala)
        colums = ["top_{}_wifi_sig".format(_i) for _i in range(top_wifi_sig_num)]
        train_wifi_matrix_df = pd.DataFrame(train_matrix[:, :top_wifi_sig_num], columns=colums, index=_part_train.index)
        test_wifi_matrix_df = pd.DataFrame(test_matrix[:, :top_wifi_sig_num], columns=colums, index=_part_test.index)
        pca = PCA(n_components=pca_component_top_wifi_sig)
        pca.fit(train_wifi_matrix_df.values)
        new_top_wifi_sig_train_f = pca.transform(train_wifi_matrix_df.values)
        new_top_wifi_sig_test_f = pca.transform(test_wifi_matrix_df.values)
        f_c = ["pca_top_wifi_sig_component_{}".format(_i) for _i in range(pca_component_top_wifi_sig)]
        new_top_wifi_sig_train_df = pd.DataFrame(new_top_wifi_sig_train_f, columns=f_c,
                                                 index=train_wifi_matrix_df.index)
        new_top_wifi_sig_test_df = pd.DataFrame(new_top_wifi_sig_test_f, columns=f_c, index=test_wifi_matrix_df.index)
        train_wifi_matrix_dfs.append(new_top_wifi_sig_train_df)
        test_wifi_matrix_dfs.append(new_top_wifi_sig_test_df)

        colums = ["top_{}_wifi_sig".format(_i) for _i in range(global_top_wifi_sig_50)]
        train_wifi_matrix_dfs2.append(
                pd.DataFrame(train_matrix[:, :global_top_wifi_sig_50], columns=colums, index=_part_train.index))
        test_wifi_matrix_dfs2.append(
                pd.DataFrame(test_matrix[:, :global_top_wifi_sig_50], columns=colums, index=_part_test.index))
        # 对同一个cluster的构建wifi rank 提升不大，待考虑
        # cluster_wifi_choose_num = 20
        # cluster_labels = np.union1d(_part_train.cluster_label.unique(), _part_test.cluster_label.unique())
        # for _label in cluster_labels:
        #     _part_train_cluster = _part_train[_part_train.cluster_label == _label]
        #     _part_test_cluster = _part_test[_part_test.cluster_label == _label]
        #     all_wifi_cluster = []
        #     _part_train_cluster.basic_wifi_info.map(lambda x: get_all_wifi(x, all_wifi_cluster))
        #     _part_test_cluster.basic_wifi_info.map(lambda x: get_all_wifi(x, all_wifi_cluster))
        #     c3 = Counter(all_wifi_cluster)
        #     sorted_wifi_cluster = sorted(c3.items(), key=lambda x: -x[1])
        #     sorted_wifi_cluster = sorted_wifi_cluster[:cluster_wifi_choose_num]
        #     d = {}  # 将wifi按rank放入dict, 排名从0开始
        #     for i, w in enumerate(sorted_wifi_cluster):
        #         d[w[0]] = i
        #     use_wifi_in_wifi_rank = _part_train_cluster.basic_wifi_info.map(lambda x: use_wifi_rank(x, d))
        #     train.loc[use_wifi_in_wifi_rank.index, "cluster_use_wifi_in_wifi_rank"] = use_wifi_in_wifi_rank
        #     use_wifi_in_wifi_rank = _part_test_cluster.basic_wifi_info.map(lambda x: use_wifi_rank(x, d))
        #     test.loc[use_wifi_in_wifi_rank.index, "cluster_use_wifi_in_wifi_rank"] = use_wifi_in_wifi_rank
        #
        #     for _top in range(3):
        #         no_use_wifi_in_wifi_rank = _part_train_cluster.basic_wifi_info.map(
        #                 lambda x: no_use_wifi_rank(x, d, _top))
        #         train.loc[no_use_wifi_in_wifi_rank.index, "cluster_no_use_wifi_top{}_in_wifi_rank".format(
        #                 _top)] = no_use_wifi_in_wifi_rank
        #         no_use_wifi_in_wifi_rank = _part_test_cluster.basic_wifi_info.map(
        #                 lambda x: no_use_wifi_rank(x, d, _top))
        #         test.loc[no_use_wifi_in_wifi_rank.index, "cluster_no_use_wifi_top{}_in_wifi_rank".format(
        #                 _top)] = no_use_wifi_in_wifi_rank
    train_wifi_matrix_df = pd.concat(train_wifi_matrix_dfs)
    test_wifi_matrix_df = pd.concat(test_wifi_matrix_dfs)
    train = pd.concat([train, train_wifi_matrix_df], axis=1)
    test = pd.concat([test, test_wifi_matrix_df], axis=1)
    train_wifi_matrix_df = pd.concat(train_wifi_matrix_dfs2)
    test_wifi_matrix_df = pd.concat(test_wifi_matrix_dfs2)
    train = pd.concat([train, train_wifi_matrix_df], axis=1)
    test = pd.concat([test, test_wifi_matrix_df], axis=1)

    train = train.drop("basic_wifi_info", axis=1)
    test = test.drop("basic_wifi_info", axis=1)
    return train, test


def rank_one(train, name):
    x1 = train.groupby(name).count()["user_id"]
    x1 = zip(list(x1.index.values), list(x1.values))
    x1 = sorted(x1, key=lambda x: -x[1])
    x1 = dict([(_x[0], _i) for _i, _x in enumerate(x1)])
    return x1


def rank_one_by_sample_size(train, test, name):
    x1 = rank_one(train, name)
    return train[name].map(lambda x: x1[x]), test[name].map(lambda x: x1[x])


def rank_label_by_one(train, test, label_dict, col_name, group_by_name, top=5):
    """

    :param train:
    :param test:
    :param label_dict:
    :param col_name: weekday,hour,
    :param group_by_name: shop_id,category_id
    :return:
    """
    weekdays = train[col_name].unique()
    tops = []
    indexs = []
    for _weekday in weekdays:
        x1 = train[train[col_name] == _weekday].groupby(group_by_name).count()["user_id"]
        x1 = zip(list(x1.index.values), list(x1.values))
        x1 = sorted(x1, key=lambda x: -x[1])
        x1 = x1[:top]
        indexs.append(_weekday)
        x1 = [label_dict[_x[0]] for _x in x1]
        x1 = np.asarray(x1)
        if x1.shape[0] < top:
            x1 = np.concatenate([x1,np.asarray([len(label_dict) for _ in range(x1.shape[0],top)])])
        tops.append(x1)
    tops = np.vstack(tops)
    indexs = np.vstack(indexs)
    r = np.concatenate([indexs, tops], axis=1)
    columns = [col_name] + ["top_{}_by_{}_group_by_{}".format(_i, col_name, group_by_name) for _i in range(top)]
    df = pd.DataFrame(r, columns=columns)
    new_train = pd.merge(train, df, on=col_name, how="left")
    new_test = pd.merge(test, df, on=col_name, how="left")
    columns.remove(col_name)
    new_train.index = train.index
    new_test.index = test.index
    return new_train[columns], new_test[columns]


def preprocess_time(train, test):
    mall_ids = train.mall_id.unique()
    new_trains = []
    new_tests = []
    for _mall_id in mall_ids:
        part_train = train[train.mall_id == _mall_id]
        part_test = test[test.mall_id == _mall_id]

        # weekday rank by sample size
        rank_train, rank_test = rank_one_by_sample_size(part_train, part_test, "hour")
        train.loc[part_train.index, "hour_rank_by_sample"] = rank_train
        test.loc[part_test.index, "hour_rank_by_sample"] = rank_test
        # hour rank by sample size
        rank_train, rank_test = rank_one_by_sample_size(part_train, part_test, "weekday")
        train.loc[part_train.index, "weekday_rank_by_sample"] = rank_train
        test.loc[part_test.index, "weekday_rank_by_sample"] = rank_test

        # 按照shop 进行sample rank label
        label_dict = rank_one(part_train, "shop_id")
        new_train, new_test = rank_label_by_one(part_train, part_test, label_dict, "weekday", "shop_id")
        new_train2, new_test2 = rank_label_by_one(part_train, part_test, label_dict, "hour", "shop_id")
        new_train = pd.concat([new_train, new_train2], axis=1)
        new_test = pd.concat([new_test, new_test2], axis=1)

        label_dict = rank_one(part_train, "category_id")
        new_train2, new_test2 = rank_label_by_one(part_train, part_test, label_dict, "weekday", "category_id")
        new_train = pd.concat([new_train, new_train2], axis=1)
        new_test = pd.concat([new_test, new_test2], axis=1)
        new_train2, new_test2 = rank_label_by_one(part_train, part_test, label_dict, "hour", "category_id")
        new_train = pd.concat([new_train, new_train2], axis=1)
        new_test = pd.concat([new_test, new_test2], axis=1)
        new_trains.append(new_train)
        new_tests.append(new_test)
    new_train = pd.concat(new_trains)
    new_test = pd.concat(new_tests)
    train = pd.concat([train, new_train], axis=1)
    test = pd.concat([test, new_test], axis=1)

    return train, test


def preprocess(mall_id=""):
    train = load_train()
    test = load_testA()

    if (mall_id != ""):
        train = train[train.mall_id == mall_id]
        test = test[test.mall_id == mall_id]

    train = preprocess_basic_wifi(train)
    test = preprocess_basic_wifi(test)

    train = preprocess_basic_time(train)
    test = preprocess_basic_time(test)

    # train, test = preprocess_time(train, test) # 效果变差
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
        df = pd.DataFrame(
                {"wifi_name": [wifi[0] for wifi in sorted_wifi], "wifi_num": [wifi[1] for wifi in sorted_wifi]})
        df.index.name = "wifi_rank"
        df.to_csv("../data/wifi_info_cache/{}_rank.csv".format(_mall_id))
        d = rank_sorted_wifi(sorted_wifi)
        for _part_data, name in zip(part_datas, names):
            wifi_matrix = np.zeros((_part_data.shape[0], len(sorted_wifi)))
            # wifi_matrix[:] = -115
            use_wifi_str = []
            _part_data.loc[:, "i_index"] = range(_part_data.shape[0])
            _part_data[["basic_wifi_info", "i_index"]].apply(
                    lambda x: basic_wifi_map2matrix(x, wifi_matrix, d, use_wifi_str), axis=1)
            a = np.asarray(use_wifi_str)
            # 用csv 存读取很慢, 将index, usewifi 和matrix 分开存
            np.save("../data/wifi_info_cache/{}_{}_index".format(name, _mall_id), _part_data.index)
            np.save("../data/wifi_info_cache/{}_{}_use_wifi".format(name, _mall_id), a)
            # 用稀疏矩阵存取
            x = sp.csc_matrix(wifi_matrix)
            scipy.save("../data/wifi_info_cache/{}_{}_matrix".format(name, _mall_id), x)

            # !!!有wifi同名情况


def get_wifi_cache(mall_id):
    df = pd.read_csv("../data/wifi_info_cache/{}_rank.csv".format(mall_id))
    train_index = scipy.load("../data/wifi_info_cache/{}_{}_index.npy".format("train", mall_id))
    test_index = scipy.load("../data/wifi_info_cache/{}_{}_index.npy".format("test", mall_id))
    train_use_wifi = scipy.load("../data/wifi_info_cache/{}_{}_use_wifi.npy".format("train", mall_id))
    test_use_wifi = scipy.load("../data/wifi_info_cache/{}_{}_use_wifi.npy".format("test", mall_id))
    train_matrix = scipy.load("../data/wifi_info_cache/{}_{}_matrix.npy".format("train", mall_id))[()].toarray()
    test_matrix = scipy.load("../data/wifi_info_cache/{}_{}_matrix.npy".format("test", mall_id))[()].toarray()
    train_matrix[train_matrix[:] == 0] = -115
    test_matrix[test_matrix[:] == 0] = -115
    return df, (train_index, train_use_wifi, train_matrix), (test_index, test_use_wifi, test_matrix)


def do_wifi_cache():
    wifi_info2csv([load_train(), load_testA()], ["train", "test"])


def wifi_sig_feature_names(mall_id):
    info = load_shop_info()
    shop_size = info[info.mall_id == mall_id].shape[0]
    top_wifi_sig_num = global_top_wifi_sig_num
    if isinstance(top_wifi_sig_num, str):
        scala = float(top_wifi_sig_num.split("_")[2])
        top_wifi_sig_num = int(shop_size * scala)
    f = ["top_{}_wifi_sig".format(_i) for _i in range(top_wifi_sig_num)]
    return f


if __name__ == '__main__':
    # save_result(None, "../result/test", ["a", "b"])
    do_wifi_cache()
