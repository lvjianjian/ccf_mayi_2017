# encoding = utf-8
from ..common import *
import numpy as np
import pandas as pd
# sys.append("../")
from ..datasets import *
class features_extractors(object):
	# set of feature extractors
    def __init__(self, config, shop_info, test_data, train_data):
        self.features_prefix = config.features_prefix
        self.features_extractors = config.features_extractors
        self.features_save = config.features_save
        self.shop_info = shop_info()
        self.test_data = evaluation_info()
        self.train_data = usr_behavior_info()
        
    def extract(self):
        features_group = []
        for k, v in features_extractors:
            cmd = 'features = %s(v)' %k
            exec(cmd)
            # features_group.append(features)
        features = np.concatenate(features_group, axis=1)


    def extract_wifi_features(self, parameters):
        # The type of the parameters if dictionary
        train, test = preprocess(mall_id = parameters["mall_id"])
        # if parameters["wifi_info_cache_pre_fix"]



		
