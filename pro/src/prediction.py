# encoding=utf-8
from datasets import *
from utils import *
import yaml
import argparse
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser(description="predict program for Ant shops")
parser.add_argument("--config", type=str, help="the config path")
args = parser.parse_args()

config = config_object(args.config)
model_params = config.model_params
extractors_param = config.features_param
usr_behavior_data = usr_behavior_info().data
shop_data = shop_info().data
evaluation_data = evaluation_info().data
mall_ids = shop_data.mall_id.unique()
# extractors = config.features_param
features_extractor = features_extractors(extractors_param, shop_data,
                                        evaluation_data, usr_behavior_data)
# the phase of train and test are included in the class of predictor 
# predictor = predictors()
offline_predicts, offline_reals = [], []
online_predicts, online_rowids = {}, {}
for _ in range(model_params["kfold"]):
    offline_predicts.append({})
    offline_reals.append({})

for i, mall_id in enumerate(mall_ids):
    print("{}/{}".format(i, len(mall_ids)))
    shops = shop_data[shop_data.mall_id == mall_id].shop_id.unique()
    label_encoder = LabelEncoder().fit(shops)
    num_class = len(shops)
    model_params["hyperparams"]["num_class"] = num_class

    train_features, y , test_features =  features_extractor.extract(mall_id)
    predict_func = predictors(train_features, y, test_features, model_params, label_encoder)
    if model_params["offline"]:
        predicts, reals = predict_func.predictor(mall_id, model_params)
        for k in range(model_params["kfold"]):
            offline_predicts[k][mall_id] = predicts[k]
            offline_reals[k][mall_id] = predicts[k]
    else:
        off_predicts, off_reals, on_predicts, on_rowids = predict_func.predictor(mall_id, model_params)
        online_predicts[mall_id] = on_predicts
        online_rowids[mall_id] = on_rowids
        for k in range(model_params["kfold"]):
            offline_predicts[k][mall_id] = predicts[k]
            offline_reals[k][mall_id] = predicts[k]
result = {}
for _mall_id in mall_ids:
    accs = []
    for _index in range(model_params["kfold"]):
        _acc = acc(offline_predicts[_index][_mall_id], offline_reals[_index][_mall_id])
        accs.append(_acc)
    print _mall_id + "'s acc is", np.mean(accs)
    result[_mall_id] = np.mean(accs)
accs = []
for _index in range(model_params["kfold"]):
    all_predict = np.concatenate(offline_reals[_index].values())
    all_true = np.concatenate(offline_predicts[_index].values())
    _acc = acc(all_predict, all_true)
    accs.append(_acc)
print "all acc is", np.mean(accs)
result["all_acc"] = np.mean(accs)
path = args.file_prefix + "_offline_acc"
save_acc(result, path, None)

if not offline:
    all_rowid = np.concatenate(all_rowid.values())
    all_predict = np.concatenate(all_predict.values())
    result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
    result.sort_values(by="row_id", inplace=True)
    path = args.file_prefix + "_online_result"
    save_result(result, path, None)








