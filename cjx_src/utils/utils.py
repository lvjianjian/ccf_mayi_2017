# encoding=utf-8
# this file is for collecting the tool functionsdef 

import pandas as pd 
from datetime import datetime
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
from 

def read_behavior_info( f="../data/train-ccf_first_round_user_shop_behavior.csv" ):
    #  return the csv object of usr_shop_behavior
    behavior_info = pd.read_csv(f)
    return behavior_info

def read_evaluation_info( f="../data/AB-test-evaluation_public.csv"):
    # test-evaluation_public info
    evaluation_info = pd.read_csv(f)
    return evaluation_info

def read_shop_info( f="../data/train-ccf_first_round_shop_info.csv" ):
    # shop_info 
    shop_info = pd.read_csv(f)
    shop_info.columns = ['shop_id', 'category_id', 'shop_longitude',
                         'shop_latitude', 'price', 'mall_id']
    return shop_info

def merge_behavior_shop(behavior_info, shop_info):
    # merge the behavior info and shop info
    # return the merged information
    train_info = pd.merge(behavior_info, shop_info, on="shop_id", how="left")
    return train_info

def process_time(data_info):
    # dataframe information
    # shop_info.rename(columns={'longitude': 'longitude_shop', 'latitude': 'latitude_shop'}, inplace=True)
    data_info['row_id'] = range(500000, 500000+len(train))

    # transform the str type to datetime type
    data_info['time_stamp'] = pd.to_datetime(train['time_stamp'])

    data_info['day'] = data_info.time_stamp.dt.dayofyear
    first_day = data_info.day.min()
    data_info['day'] =  data_info.day - first_day
    data_info['hour'] = data_info.time_stamp.dt.hour
    data_info['dayofweek'] = data_info.time_stamp.dt.dayofweek
    return data_info
