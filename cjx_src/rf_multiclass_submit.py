# coding: utf-8
import numpy as np
import pandas as pd
import scipy as sp
from sklearn.model_selection import train_test_split
from itertools import product
from collections import Counter
from datetime import datetime
import warnings
import sys
import heapq
import math
import os
import gc
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

train_path = '../data/train-ccf_first_round_user_shop_behavior.csv'
shop_info_path = "../data/train-ccf_first_round_shop_info.csv"
test_path = "../data/AB-test-evaluation_public.csv"
load = True
cache_path = 'ali_cache_tree/'
if not os.path.exists(cache_path):
    os.mkdir(cache_path)
    
t0 = time.time()

if load == True:
    train = pd.read_csv(train_path)
    shop_info = pd.read_csv(shop_info_path)
    shop_info.rename(columns={'longitude': 'longitude_shop', 'latitude': 'latitude_shop'}, inplace=True)
    train['row_id'] = range(500000, 500000+len(train))

    train['time_stamp'] = pd.to_datetime(train['time_stamp'])

    train['day'] =  train.time_stamp.dt.dayofyear
    first_day = train.day.min()
    train['day'] =  train.day - first_day
    train['hour'] = train.time_stamp.dt.hour
    train['dayofweek'] = train.time_stamp.dt.dayofweek
    train = pd.merge(train, shop_info, on='shop_id', how='left')

    
    test = pd.read_csv(test_path)
    test['time_stamp'] = pd.to_datetime(test['time_stamp'])
    test['day'] =  test.time_stamp.dt.dayofyear - first_day
    test['hour'] =  test.time_stamp.dt.hour
    test['dayofweek'] = test.time_stamp.dt.dayofweek


    common_columns = ['row_id', 'user_id', 'mall_id', 'time_stamp', 'longitude', 'latitude', \
                      'wifi_infos', 'day', 'hour', 'dayofweek']
    train_all = pd.concat([ train[common_columns], test[common_columns]], axis=0)
    
def get_wifi_tree_rec_shops(end_days, top_num, n_thread):
    train_records = train[(train.day<end_days[0])][['row_id', 'mall_id', 'shop_id' , 'wifi_infos']]
    test_records = train_all[ train_all.day.isin(end_days)][['row_id', 'mall_id', 'wifi_infos']]
    
    knn_rec_shops = []
    mall_ids = train_records.mall_id.unique()
    # mall_ids = ["m_6803", "m_3313"]
    
    def get_mall_tree_rec_shops(mall_id):
        print (mall_id)
        mall_tree_rec_shops = []
        train_records_in_mall = train_records[train_records.mall_id==mall_id]
        test_records_in_mall = test_records[test_records.mall_id==mall_id]
        row_shop_map = dict(zip(train_records_in_mall.row_id.values, train_records_in_mall.shop_id.values))
        rowid_wifi_strength = []
        row_ids, bssids, strengths, connects = [], [], [], []
        train_mall_wifiset = set()

        for x in train_records_in_mall.values:
            row_id, wifi_infos = x[0], x[3]
            for wifi_info in wifi_infos.split(';'):
                info = wifi_info.split('|')
                if info[0] not in train_mall_wifiset:
                    train_mall_wifiset.add(info[0])
                row_ids.append(int(row_id))
                bssids.append(info[0])
                strengths.append(int(info[1]))
        min_strength = np.min(strengths) - 1        

        for x in test_records_in_mall.values:
            row_id, wifi_infos = x[0], x[2]
            for wifi_info in wifi_infos.split(';'):
                info = wifi_info.split('|')
                if info[0] in train_mall_wifiset:
                    row_ids.append(int(row_id))
                    bssids.append(info[0])
                    strengths.append(int(info[1]))

        strengths = map(lambda x: x-min_strength, strengths)
        row_wifi_strength = pd.DataFrame({'row_id':row_ids,'bssid': bssids,'strength':strengths})
        row_wifi_strength = pd.pivot_table(row_wifi_strength, index=['row_id'], columns=['bssid'], values=['strength']).fillna(0)['strength'] 
        
        train_row_wifi_strength = row_wifi_strength[row_wifi_strength.index.isin(train_records_in_mall.row_id)]
        train_row_wifi_strength =  pd.merge(train_row_wifi_strength, train_all[['row_id', 'longitude', 'latitude', 'hour', 'dayofweek']], left_index=True, right_on='row_id', how='left')
        train_label = train_row_wifi_strength.row_id.map(row_shop_map)
        train_row_wifi_strength.drop('row_id', axis=1, inplace=True)
        
        le = LabelEncoder()
        train_label = le.fit_transform(train_label)
        
        test_row_wifi_strength = row_wifi_strength[row_wifi_strength.index.isin(test_records_in_mall.row_id)]
        test_row_wifi_strength =  pd.merge(test_row_wifi_strength,train_all[['row_id', 'longitude', 'latitude', 'hour', 'dayofweek']], left_index=True, right_on='row_id', how='left')
        test_row_ids = test_row_wifi_strength.row_id.values
        test_row_wifi_strength.drop('row_id', axis=1, inplace=True)        
        
        rf = RandomForestClassifier(n_estimators=220, n_jobs=n_thread)
        rf.fit(train_row_wifi_strength, train_label)
        test_label = le.inverse_transform(rf.predict(test_row_wifi_strength))
        all_tree_rec_shops  = pd.DataFrame(zip(test_row_ids,  test_label), columns=['row_id', 'shop_id'])
        return all_tree_rec_shops
    
    tree_rec_shops = map(get_mall_tree_rec_shops, mall_ids)
    tree_rec_shops = pd.concat(tree_rec_shops, axis=0)
    return  tree_rec_shops    

results = get_wifi_tree_rec_shops(list(range(31, 31+14)), 1, 4)
test = pd.read_csv(test_path)
results = pd.merge(test[['row_id']], results, on='row_id', how='left')
results.fillna('0',inplace=True)
results.to_csv('./submit/test_rf.csv', index = None)

print('一共用时{}秒'.format(time.time()-t0))
print (datetime.now())