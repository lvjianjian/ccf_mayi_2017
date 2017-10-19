#!/usr/bin/env python
#encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-18, 22:17

@Description:

@Update Date: 17-10-18, 22:17
"""
from util import *
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

from sklearn.decomposition import PCA

pca_test = True
pca_components = 15
def main(offline=False, mall_id ="m_690"):  # 每个mall 训练一个分类器
    model_name = "xgboost"

    train, test = preprocess(mall_id)

    # 划分验证 测试集
    if offline:
        train, valid, test = train_split(train)
    else:
        train, valid, _ = train_split(train, ratio=(9, 1, 0))

    mall_ids = train.mall_id.unique()

    xgb_models = {}
    label_encoders = {}

    # 模型参数
    objective = "multi:softmax"
    eval_metric = "merror"
    eta = 0.02
    max_depth = 10
    subsample = 0.8
    colsample_bytree = 0.8
    min_child_weight = 5
    gamma = 1
    alpha = 0
    _lambda = 0
    n_round = 100
    early_stop_rounds = 8

    # 特征选择
    def remove_f(features, fname):
        if fname in features:
            features.remove(fname)

    features = list(test.columns)
    remove_f(features, "user_id")
    remove_f(features, "mall_id")
    remove_f(features, "time_stamp")
    remove_f(features, "wifi_infos")
    remove_f(features, "dt")
    remove_f(features, "shop_id")
    remove_f(features, "row_id")
    remove_f(features, "category_id")
    remove_f(features, "shop_longitude")
    remove_f(features, "shop_latitude")
    remove_f(features, "price")

    # features = wifi_sig_feature_names(mall_id)

    print "features size", len(features)
    print features

    # 为每个商场构建一个分类器
    # for _index, _mall_id in enumerate(mall_ids):
    _mall_id = mall_id
    print "train: ", _mall_id, " {}/{}".format(0, len(mall_ids))
    _train = train[train.mall_id == _mall_id]
    _valid = valid[valid.mall_id == _mall_id]
    all_shop_id = np.union1d(_train.shop_id.unique(), _valid.shop_id.unique())
    label_encoder = LabelEncoder().fit(all_shop_id)
    num_class = len(all_shop_id)
    print "shop size", num_class

    # PCA 测试
    pca = PCA(n_components=pca_components,whiten=False)

    _train_x = _train[features].values
    if pca_test:
        pca.fit(_train_x)
        _train_x = pca.transform(_train_x)
    _train_y = label_encoder.transform(_train["shop_id"].values)
    _valid_x = _valid[features].values
    if pca_test:
        _valid_x = pca.transform(_valid_x)
    _valid_y = label_encoder.transform(_valid["shop_id"].values)
    _train = xgb.DMatrix(_train_x, label=_train_y)
    _valid = xgb.DMatrix(_valid_x, label=_valid_y)
    evals = [(_train, "train"), (_valid, "valid")]
    bst = xgb.train({
        "objective": objective,
        "eval_metric": eval_metric,
        "eta": eta,
        "max_depth": max_depth,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "alpha": alpha,
        "lambda": _lambda,
        "num_class": num_class,
        "silent": 1,
    },
            _train,
            n_round,
            evals=evals,
            early_stopping_rounds=early_stop_rounds)

    ####################################
    # 将train和valid合并训练 TO-DO
    ####################################


    xgb_models[_mall_id] = bst
    label_encoders[_mall_id] = label_encoder

    # 进行预测
    predicts = {}
    row_ids_or_true = {}
    # for _mall_id in mall_ids:
    bst = xgb_models[_mall_id]
    label_encoder = label_encoders[_mall_id]
    _test = test[test.mall_id == _mall_id]
    _test_x = _test[features].values
    if pca_test:
        _test_x = pca.transform(_test_x)
    _test_DMatrix = xgb.DMatrix(_test_x)
    predict = bst.predict(_test_DMatrix, ntree_limit=bst.best_iteration).astype(int)
    predict = label_encoder.inverse_transform(predict)
    predicts[_mall_id] = predict
    if offline:
        _test_y_origin = _test["shop_id"].values
        row_ids_or_true[_mall_id] = _test_y_origin
    else:
        row_ids_or_true[_mall_id] = _test["row_id"].values

    # 保存结果
    all_predict = np.concatenate(predicts.values())
    if offline:
        result = {}
        for _mall_id in mall_ids:
            _acc = acc(predicts[_mall_id], row_ids_or_true[_mall_id])
            print _mall_id + "'s acc is", _acc
            result[_mall_id] = _acc
        all_true = np.concatenate(row_ids_or_true.values())
        _acc = acc(all_predict, all_true)
        print "all acc is", _acc
        result["all_acc"] = _acc
        exit(1)
        path = "../result/offline/{}_f{}_eta{}_md{}_ss{}_csb{}_mcw{}_ga{}_al{}_la{}_mallid{}".format(model_name, len(features),
                                                                                            eta, max_depth,
                                                                                            subsample, colsample_bytree,
                                                                                            min_child_weight, gamma,
                                                                                            alpha, _lambda, mall_id)
        save_acc(result, path, features)
    else:
        all_rowid = np.concatenate(row_ids_or_true.values())
        result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
        result.sort_values(by="row_id", inplace=True)
        path = "../result/online/{}_f{}_eta{}_md{}_ss{}_csb{}_mcw{}_ga{}_al{}_la{}_mallid{}".format(model_name, len(features),
                                                                                           eta, max_depth,
                                                                                           subsample, colsample_bytree,
                                                                                           min_child_weight, gamma,
                                                                                           alpha, _lambda, mall_id)
        save_result(result, path, features)


if __name__ == '__main__':
    main(offline=True, mall_id="m_6803")
    # main(offline=False)