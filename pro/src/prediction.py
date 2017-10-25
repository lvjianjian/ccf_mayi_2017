# encoding = utf-8
from datasets import *
from utils import *

import yaml
import argparse
parser = argparse.ArgumentParser(description="predict program for Ant shops")
parser.add_argment("--config", type=str, help="the config path")
args = parser.parser_args()

config = config_object(args.config)
model_params = config.model_params
extractors_param = config.features_extractors
usr_behavior_data = usr_behavior_info().data
shop_data = shop_info().data
evaluation_data = evaluation_data().data
features_extractor = features_extractors(extractors, shop_data, 
                                            evaluation_data, usr_behavior_data)

for mall_id in shop_data.mall_id.unique():
    features_extractors.extract(mall_id)



