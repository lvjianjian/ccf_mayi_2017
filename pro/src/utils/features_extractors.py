# encoding=utf-8 
import common
import numpy as np
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.decomposition import PCA
import lightgbm as lgb
sys.path.append("../")
from datasets import *
import os
class features_extractors(object):
	# set of feature extractors
    def __init__(self, config, shop_data, evaluation_data, usr_behavior_data):
        self.features_extractors = config["features_extractors"]
        self.shop_data = shop_data
        self.evaluation_data = evaluation_data
        self.usr_behavior_data = usr_behavior_data
        self.mall_ids  = self.shop_data.mall_id.unique()
        self.merged_data = common.merge_behavior_shopinfo(usr_behavior_data,
                                                shop_data)
    def extract(self, mall_id):
        train_features_group = []
        test_features_group = []
        print(self.features_extractors)
        for (k, v) in self.features_extractors.items():
            cmd = 'train_features, train_y, test_features = self.%s(mall_id, v)' % k
            # print(cmd)
            exec(cmd)
            # exit()
            train_features_group.append(train_features)
            test_features_group.append(test_features)
        train_features = np.concatenate(train_features_group, axis=1)
        test_features = np.concatenate(test_features_group, axis=1)
        
        return train_features, train_y, test_features

    def extract_wifi_longitude_latitude_features(self, mall_id, parameters):
        """
            parameters:
                name:
                cache_prefix:
        """

        if os.path.exists(os.path.join(parameters["cache_prefix"],
                        "{}_{}_index.npy".format("train", mall_id))
                        )==False:
            do_wifi_cache(parameters["cache_prefix"])

        model_name = "lightgbm_wifi_sig_lonlat"
        all_predict = {}
        row_ids_or_true = {}
        shop_data = self.shop_data

        train_all = self.merged_data
        test_all = self.evaluation_data
        shops = shop_data[shop_data.mall_id == mall_id].shop_id.unique()
        num_class = len(shops)
        df, train_cache, test_cache = common.get_wifi_cache(mall_id,
                                                    parameters["cache_prefix"]
                                                    )
        train_matrix = train_cache[2]
        test_matrix = test_cache[2]

        train_matrix = np.tile(-train_matrix.max(axis=1, keepdims=True),
                                (1, train_matrix.shape[1])) + train_matrix
        test_matrix = np.tile(-test_matrix.max(axis=1, keepdims=True),
                                (1, test_matrix.shape[1])) + test_matrix

        
        # wifi rank info
        train = train_all[train_all.mall_id == mall_id]

        test = test_all[test_all.mall_id == mall_id]
        common.preprocess_basic_wifi(train)
        common.preprocess_basic_wifi(test)
        sorted_wifi = common.get_sorted_wifi([train, test])
        d = common.rank_sorted_wifi(sorted_wifi)
        other_train_wifi_features = []
        other_test_wifi_features = []
        test_use_wifi_in_wifi_rank, train_use_wifi_in_wifi_rank = common.use_wifi_in_wifi_rank(test, train, d)
        other_train_wifi_features.append(
            train_use_wifi_in_wifi_rank.values.reshape((-1, 1))
            )
        other_test_wifi_features.append(
            test_use_wifi_in_wifi_rank.values.reshape((-1, 1))
            )
        # print train_use_wifi_in_wifi_rank
        for _top in range(10):
            test_no_use_wifi_in_wifi_rank, train_no_use_wifi_in_wifi_rank = common.no_use_wifi_in_wifi_rank(test,
                                                                                                     train,
                                                                                                     d,
                                                                                                     _top)
            other_train_wifi_features.append(
                train_no_use_wifi_in_wifi_rank.values.reshape((-1, 1))
                )
            other_test_wifi_features.append(
                test_no_use_wifi_in_wifi_rank.values.reshape((-1, 1))
                )

        other_train_wifi_feature = np.concatenate(
                                other_train_wifi_features, 
                                axis=1)
        other_test_wifi_feature = np.concatenate(
                                other_test_wifi_features, 
                                axis=1)
        scala = parameters["scala"]
        pca = PCA(n_components=int(num_class * scala)).fit(train_matrix)
        train_matrix = pca.transform(train_matrix)
        test_matrix = pca.transform(test_matrix)

        test_index = test_cache[0]
        label_encoder = LabelEncoder().fit(shops)

        y = label_encoder.transform(train.shop_id)

        # distance_matrix
        train_lonlats = train[["longitude", "latitude"]].values
        test_lonlats = test[["longitude", "latitude"]].values
        d = common.rank_one(train, "shop_id")
        verctors = []
        for _s, _i in d.items():
            _shop = shop_data[shop_data.shop_id == _s][["shop_longitude",
                                                "shop_latitude"]].values
            _shop = np.tile(_shop, (train_lonlats.shape[0], 1))
            verctors.append(
                    common.haversine(train_lonlats[:, 0],
                    train_lonlats[:, 1],
                    _shop[:, 0],
                    _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        distance_matrix = np.concatenate(verctors, axis=1)

        verctors = []
        for _s, _i in d.items():
            _shop = shop_data[shop_data.shop_id == _s][["shop_longitude",
                                                        "shop_latitude"]].values
            _shop = np.tile(_shop, (test_lonlats.shape[0], 1))
            verctors.append(
                    common.haversine(test_lonlats[:, 0],
                            test_lonlats[:, 1], 
                            _shop[:, 0],
                            _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        test_dis_matrix = np.concatenate(verctors, axis=1)

        pca_dis = PCA(n_components=int(round(num_class / 5))).fit(distance_matrix)
        distance_matrix = pca_dis.transform(distance_matrix)
        test_dis_matrix = pca_dis.transform(test_dis_matrix)
        train_dis_matrix = distance_matrix

        train_matrix = np.concatenate([train_matrix, train_dis_matrix], axis=1)
        test_matrix = np.concatenate([test_matrix, test_dis_matrix], axis=1)
        
        return train_matrix, y, test_matrix



		
