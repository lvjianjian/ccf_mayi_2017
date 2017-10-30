#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-26, 22:35

@Description:

@Update Date: 17-10-26, 22:35
"""

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from util import *
import os, yaml
import lightgbm as lgb
from hyperopt import fmin, hp, space_eval, tpe
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
# from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection._split import check_cv

class Stacking(object):
    def __init__(self, base_clses, meta_cls, use_prob=False, kfold=-1, stratify=True, Kfold_shuffle=False, num_class = None):
        """

        :param base_clses: sklearn model 或者lightgbm lgb的话存入tuple("lgb", params)
        :param meta_cls:
        :param objective:
        """
        self.base_clses = base_clses
        self.meta_cls = meta_cls
        self.spetial_base_clses = {}
        self.use_prob = use_prob
        self.kfold = kfold
        self.Kfold_shuffle = Kfold_shuffle
        self.stratify = stratify
        self.num_class = num_class
        if self.kfold > 0:
            self.cv = True
            self.sklearn_models = {}
            self.spetial_base_clses_params = {}
            for _cls in base_clses:
                if not isinstance(_cls, tuple):
                    base_name = type(_cls).__name__.lower()
                    for _i in range(self.kfold):
                        self.sklearn_models[base_name + "_{}".format(_i)] = clone(_cls)
                else:
                    if _cls[0].startswith("lgb"):
                        for _i in range(self.kfold):
                            lgb_name = _cls[0] + "_" + str(_i)
                            self.spetial_base_clses_params[lgb_name] = (lgb_name, _cls[1].copy())

        else:
            self.cv = False

    def get_best_iteration(self, name):
        it = []
        for _name,_bst in self.spetial_base_clses.items():
            if _name.startswith(name):
               it.append(_bst.best_iteration)
        return int(np.mean(it))


    def _lgb_train(self, _cls, train_x, train_y, valid_x=None, valid_y=None):
        _train = lgb.Dataset(train_x, label=train_y)
        _valid = None
        if valid_x is not None:
            _valid = lgb.Dataset(valid_x, label=valid_y, reference=_train)
        if _valid is None:
            bst = lgb.train(_cls[1],
                            _train)
        else:
            bst = lgb.train(_cls[1],
                            _train,
                            valid_sets=_valid)
        assert _cls[0] not in self.spetial_base_clses.keys()
        self.spetial_base_clses[_cls[0]] = bst
        return self._lgb_predict(_cls[0], train_x, self.use_prob)

    def _lgb_predict(self, _cls_name, x, use_proba=False):
        bst = self.spetial_base_clses[_cls_name]
        if use_proba:
            p = bst.predict(x, bst.best_iteration)
        else:
            p = np.argmax(bst.predict(x, bst.best_iteration), axis=1).astype(int).reshape((-1, 1))
        return p

    def _sklearn_predict(self, model, x, use_proba=False):
        if use_proba:
            p = model.predict_proba(x)
        else:
            p = model.predict(x).reshape((-1, 1))

        if use_proba and self.num_class is not None:#针对没有出现的类别导致的feature不一样
            sample_size = p.shape[0]
            clss = model.classes_
            p = dict(zip(clss,p.transpose()))
            for _i in range(self.num_class):
                if _i not in p.keys():
                    p[_i] = np.zeros((sample_size,))
            p = np.vstack(p.values()).transpose()
        return p

    def fit(self, train_x, train_y, valid_x=None, valid_y=None):
        self.split = check_cv(self.kfold, train_y, self.stratify)
        self.split.shuffle = self.Kfold_shuffle
        p_rs = []
        v_rs = []
        if not self.cv:
            for _cls in self.base_clses:
                if isinstance(_cls, tuple):
                    if _cls[0].startswith("lgb"):
                        p = self._lgb_train(_cls, train_x, train_y, valid_x, valid_y)
                        p_rs.append(p)
                        if valid_x is not None:
                            p = self._lgb_predict(_cls[0], valid_x, self.use_prob)
                            v_rs.append(p)
                else:
                    _cls.fit(train_x, train_y)
                    p = self._sklearn_predict(_cls, train_x, self.use_prob)
                    p_rs.append(p)
                    if valid_x is not None:
                        p = self._sklearn_predict(_cls, valid_x, self.use_prob)
                        v_rs.append(p)

        else:
            for _cls in self.base_clses:
                p_part = []
                y_index = []
                v_part = []
                if isinstance(_cls, tuple):  # lgb or xgb
                    if _cls[0].startswith("lgb"):
                        for _i, (_train_index, _test_index) in enumerate(self.split.split(train_x, train_y)):
                            _train_x = train_x[_train_index]
                            _train_y = train_y[_train_index]
                            _test_x = train_x[_test_index]

                            lgb_name = _cls[0] + "_" + str(_i)
                            self._lgb_train(self.spetial_base_clses_params[lgb_name],
                                            _train_x,
                                            _train_y,
                                            valid_x,
                                            valid_y)
                            p_part.append(self._lgb_predict(lgb_name, _test_x, self.use_prob))
                            y_index.append(_test_index.reshape((-1, 1)))
                            if valid_x is not None:
                                p = self._lgb_predict(lgb_name, valid_x, self.use_prob)
                                v_part.append(p)

                else:  # sklearn
                    for _i, (_train_index, _test_index) in enumerate(self.split.split(train_x, train_y)):
                        _train_x = train_x[_train_index]
                        _train_y = train_y[_train_index]
                        _test_x = train_x[_test_index]
                        name = type(_cls).__name__.lower()
                        model = self.sklearn_models[name + "_" + str(_i)]
                        model.fit(_train_x, _train_y)
                        p = self._sklearn_predict(model, _test_x, self.use_prob)
                        p_part.append(p)
                        y_index.append(_test_index.reshape((-1, 1)))
                        if valid_x is not None:
                            p = self._sklearn_predict(model, valid_x, self.use_prob)
                            v_part.append(p)
                print p_part[0].shape, p_part[1].shape
                p_part = np.vstack(p_part)
                y_index = np.vstack(y_index)
                p = dict(zip(list(y_index.reshape((-1,))), list(p_part)))
                p_rs.append(np.vstack(p.values()))
                if valid_x is not None:
                    v_rs.append(np.mean(np.stack(v_part, axis=2), axis=2))

        new_train = np.hstack(p_rs)

        if valid_x is not None:
            valid_x = np.hstack(v_rs)

        assert new_train.shape[0] == train_y.shape[0]
        if valid_x is not None:
            assert valid_x.shape[0] == valid_y.shape[0]
            assert new_train.shape[1] == valid_x.shape[1]

        # 次学习其
        if isinstance(self.meta_cls, tuple):
            if self.meta_cls[0].startswith("lgb"):
                self._lgb_train(self.meta_cls, new_train, train_y, valid_x, valid_y)
        else:
            self.meta_cls.fit(new_train, train_y)

    def predict(self, test_x):
        p_rs = []
        if not self.cv:
            for _cls in self.base_clses:
                if isinstance(_cls, tuple):
                    if _cls[0].startswith("lgb"):
                        p = self._lgb_predict(_cls[0], test_x, self.use_prob)
                        p_rs.append(p)
                else:
                    p = self._sklearn_predict(_cls, test_x, self.use_prob)
                    p_rs.append(p)

        else:
            for _cls in self.base_clses:
                p_part = []
                if isinstance(_cls, tuple):  # lgb or xgb
                    if _cls[0].startswith("lgb"):
                        for _i in range(self.kfold):
                            lgb_name = _cls[0] + "_" + str(_i)
                            p_part.append(self._lgb_predict(lgb_name, test_x, self.use_prob))

                else:  # sklearn
                    for _i in range(self.kfold):
                        name = type(_cls).__name__.lower()
                        model = self.sklearn_models[name + "_" + str(_i)]
                        p = self._sklearn_predict(model, test_x, self.use_prob)
                        p_part.append(p)
                p_rs.append(np.mean(np.stack(p_part, axis=2), axis=2))

        new_train = np.hstack(p_rs)
        # 次学习器
        if isinstance(self.meta_cls, tuple):
            if _cls[0].startswith("lgb"):
                p = self._lgb_predict(self.meta_cls[0], new_train)
        else:
            p = self._sklearn_predict(self.meta_cls, new_train)
        return p.reshape((-1,))


def main_leave_one_week(offline, mall_ids=-1, use_hyperopt=False, default_scala=1, use_default_scala=False):
    model_name = "stacking_leave_one_week_wifi_matrix_rank_lonlat_matrix"
    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    if mall_ids == -1:
        mall_ids = shop_info.mall_id.unique()
    offline_predicts = []
    offline_reals = []
    all_rowid = {}
    all_predicts = {}
    if os.path.exists("../data/best_scala/best_scala_{}.yaml".format(model_name)):
        best_scala = yaml.load(open("../data/best_scala/best_scala_{}.yaml".format(model_name), "r"))
    else:
        best_scala = {}
    if use_default_scala:
        best_scala = {}
    kfold = 1
    for _ in range(kfold):
        offline_predicts.append({})
        offline_reals.append({})
    for _index, mall_id in enumerate(mall_ids):
        print "train: ", mall_id, " {}/{}".format(_index, len(mall_ids))
        shops = shop_info[shop_info.mall_id == mall_id].shop_id.unique()
        num_class = len(shops)
        df, train_cache, test_cache = get_wifi_cache(mall_id)
        train_matrix = train_cache[2]
        test_matrix = test_cache[2]

        # 将wifi 信号加上每个sample的最大wifi信号， 屏蔽个体之间接收wifi信号的差异
        train_matrix = np.tile(-train_matrix.max(axis=1, keepdims=True), (1, train_matrix.shape[1])) + train_matrix
        test_matrix = np.tile(-test_matrix.max(axis=1, keepdims=True), (1, test_matrix.shape[1])) + test_matrix

        # wifi rank info
        train = train_all[train_all.mall_id == mall_id]
        test = test_all[test_all.mall_id == mall_id]
        preprocess_basic_wifi(train)
        preprocess_basic_wifi(test)
        sorted_wifi = get_sorted_wifi([train, test])
        d = rank_sorted_wifi(sorted_wifi)
        other_train_wifi_features = []
        other_test_wifi_features = []
        test_use_wifi_in_wifi_rank, train_use_wifi_in_wifi_rank = use_wifi_in_wifi_rank(test, train, d)
        other_train_wifi_features.append(train_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
        other_test_wifi_features.append(test_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
        # print train_use_wifi_in_wifi_rank
        for _top in range(10):
            test_no_use_wifi_in_wifi_rank, train_no_use_wifi_in_wifi_rank = no_use_wifi_in_wifi_rank(test,
                                                                                                     train,
                                                                                                     d,
                                                                                                     _top)
            other_train_wifi_features.append(train_no_use_wifi_in_wifi_rank.values.reshape((-1, 1)))
            other_test_wifi_features.append(test_no_use_wifi_in_wifi_rank.values.reshape((-1, 1)))

        other_train_wifi_feature = np.concatenate(other_train_wifi_features, axis=1)
        other_test_wifi_feature = np.concatenate(other_test_wifi_features, axis=1)

        test_index = test_cache[0]
        label_encoder = LabelEncoder().fit(shops)
        y = label_encoder.transform(train.shop_id)

        # distance_matrix
        # 加入经纬度 直接经纬度效果很差
        train_lonlats = train[["longitude", "latitude"]].values
        test_lonlats = test[["longitude", "latitude"]].values
        # 用户经纬度与各个商店的距离矩阵
        d = rank_one(train, "shop_id")
        verctors = []
        for _s, _index in d.items():
            _shop = shop_info[shop_info.shop_id == _s][["shop_longitude", "shop_latitude"]].values
            _shop = np.tile(_shop, (train_lonlats.shape[0], 1))
            verctors.append(
                    haversine(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        distance_matrix = np.concatenate(verctors, axis=1)

        verctors = []
        for _s, _index in d.items():
            _shop = shop_info[shop_info.shop_id == _s][["shop_longitude", "shop_latitude"]].values
            _shop = np.tile(_shop, (test_lonlats.shape[0], 1))
            verctors.append(
                    haversine(test_lonlats[:, 0], test_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
            # verctors.append(bearing(train_lonlats[:, 0], train_lonlats[:, 1], _shop[:, 0], _shop[:, 1]).reshape((-1, 1)))
        test_dis_matrix = np.concatenate(verctors, axis=1)

        # pca_dis = PCA(n_components=int(round(num_class / 2))).fit(distance_matrix)
        # distance_matrix = pca_dis.transform(distance_matrix)
        # test_dis_matrix = pca_dis.transform(test_dis_matrix)
        train_dis_matrix = distance_matrix

        # 模型参数
        num_leaves = 35
        learning_rate = 0.04
        feature_fraction = 0.7
        bagging_fraction = 0.8
        bagging_freq = 5
        n_round = 1000
        early_stop_rounds = 15

        params = {
            'task': 'train',
            'boosting_type': 'gbdt',
            'objective': 'multiclass',
            'metric': ['multi_error'],
            'num_leaves': num_leaves,
            'learning_rate': learning_rate,
            'feature_fraction': feature_fraction,
            'bagging_fraction': bagging_fraction,
            'bagging_freq': bagging_freq,
            'verbose': 1,
            'num_class': num_class,
            'early_stopping_rounds': early_stop_rounds,
            "num_round": n_round

        }

        _train_index, _valid_index = get_last_one_week_index(train)
        argsDict = {}
        if use_hyperopt:
            def objective(argsDict):
                _train_matrix = train_matrix.copy()
                _scala = argsDict["scala"]
                pca = PCA(n_components=int(num_class * _scala)).fit(_train_matrix)
                _train_matrix = pca.transform(_train_matrix)

                _train_matrix = np.concatenate([_train_matrix,
                                                train_dis_matrix,
                                                other_train_wifi_feature],
                                               axis=1)
                _train_x = _train_matrix[_train_index]
                _train_y = y[_train_index]
                _train = lgb.Dataset(_train_x, label=_train_y)

                _valid_x = _train_matrix[_valid_index]
                _valid_y = y[_valid_index]
                _valid = lgb.Dataset(_valid_x, label=_valid_y, reference=_train)

                bst = lgb.train(params,
                                _train,
                                n_round,
                                valid_sets=_valid,
                                early_stopping_rounds=early_stop_rounds)
                y_predict = np.argmax(bst.predict(_valid_x, num_iteration=bst.best_iteration), axis=1).astype(int)
                return -acc(y_predict, _valid_y)

            space = {
                "scala": hp.uniform("scala", 0.3, 8)
            }

            best_sln = fmin(objective, space, algo=tpe.suggest, max_evals=10)
            argsDict = space_eval(space, best_sln)
            best_scala[mall_id] = argsDict["scala"]
        else:
            if len(best_scala) == 0:
                argsDict["scala"] = default_scala
            else:
                argsDict["scala"] = best_scala[mall_id]

        scala = argsDict["scala"]
        print "use scala:", scala
        pca = PCA(n_components=int(num_class * scala)).fit(train_matrix)
        #pca = PCA(n_components=int(num_class * scala)).fit(np.concatenate([train_matrix, test_matrix]))
        train_matrix = pca.transform(train_matrix)
        test_matrix = pca.transform(test_matrix)

        train_matrix = np.concatenate([train_matrix,
                                       train_dis_matrix,
                                       other_train_wifi_feature],
                                      axis=1)
        test_matrix = np.concatenate([test_matrix,
                                      test_dis_matrix,
                                      other_test_wifi_feature],
                                     axis=1)

        print "num_class", num_class

        # kfold
        print "train", mall_id
        stacking_k_fold = 3

        _index = 0
        for _train_index, _valid_index in [(_train_index, _valid_index)]:
            _train_x = train_matrix[_train_index]
            _train_y = y[_train_index]

            _valid_x = train_matrix[_valid_index]
            _valid_y = y[_valid_index]

            clf1 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=1)
            clf2 = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, verbose=1)
            clf3 = ("lgb1", params)

            final_params = params.copy()
            final_params["learning_rate"] = 0.005
            final_params["early_stopping_rounds"] = 50
            clf4 = ("lgb2", final_params)
            sclf = Stacking([clf1, clf2, clf3], clf4, use_prob=True, kfold=stacking_k_fold, num_class=num_class)
            sclf.fit(_train_x,
                     _train_y,
                     _valid_x,
                     _valid_y)

            predict = sclf.predict(_valid_x)
            print predict
            print _valid_y

            predict = label_encoder.inverse_transform(predict)
            offline_predicts[_index][mall_id] = predict
            offline_reals[_index][mall_id] = label_encoder.inverse_transform(_valid_y)
            _index += 1


        if not offline:  # 线上
            base_iteration = sclf.get_best_iteration("lgb")
            meta_iteration = sclf.get_best_iteration("lgb2")
            print base_iteration
            print meta_iteration
            clf1 = RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=1)
            clf2 = ExtraTreesClassifier(n_estimators=1000, n_jobs=-1, verbose=1)

            if "early_stopping_rounds" in params:
                params.pop("early_stopping_rounds")
            params["num_round"] = base_iteration + int(base_iteration * 0.05)
            clf3 = ("lgb1", params)

            final_params = params.copy()
            if "early_stopping_rounds" in final_params:
                final_params.pop("early_stopping_rounds")
            final_params["learning_rate"] = 0.005
            final_params["num_round"] = meta_iteration + int(meta_iteration * 0.05)

            clf4 = ("lgb2", final_params)
            sclf = Stacking([clf1, clf2, clf3], clf4, use_prob=True, kfold=stacking_k_fold, num_class=num_class)
            sclf.fit(train_matrix, y)
            predict = sclf.predict(test_matrix)
            predict = label_encoder.inverse_transform(predict)
            all_predicts[mall_id] = predict
            all_rowid[mall_id] = test_all[np.in1d(test_all.index, test_index)].row_id.values

    result = {}
    for _mall_id in mall_ids:
        accs = []
        for _index in range(kfold):
            _acc = acc(offline_predicts[_index][_mall_id], offline_reals[_index][_mall_id])
            accs.append(_acc)
        print _mall_id + "'s acc is", np.mean(accs)
        result[_mall_id] = np.mean(accs)
    accs = []
    for _index in range(kfold):
        all_predict = np.concatenate(offline_reals[_index].values())
        all_true = np.concatenate(offline_predicts[_index].values())
        _acc = acc(all_predict, all_true)
        accs.append(_acc)
    print "all acc is", np.mean(accs)

    if len(mall_ids) < 50:
        exit(1)

    result["all_acc"] = np.mean(accs)
    path = "../result/offline/{}_f{}_lr{}_leaves{}_ff{}_bf{}_bfq{}_es{}".format(model_name,
                                                                                "num_class_{}".format(scala),
                                                                                learning_rate,
                                                                                num_leaves,
                                                                                feature_fraction,
                                                                                bagging_fraction,
                                                                                bagging_freq,
                                                                                early_stop_rounds)
    save_acc(result, path, None)

    if use_hyperopt:
        yaml.dump(best_scala, open("../data/best_scala/best_scala_{}.yaml".format(model_name), "w"))

    if not offline:
        all_rowid = np.concatenate(all_rowid.values())
        all_predict = np.concatenate(all_predicts.values())
        result = pd.DataFrame(data={"row_id": all_rowid, "shop_id": all_predict})
        result.sort_values(by="row_id", inplace=True)
        path = "../result/online/{}_f{}_lr{}_leaves{}_ff{}_bf{}_bfq{}_es{}".format(model_name,
                                                                                   "num_class_{}".format(scala),
                                                                                   learning_rate,
                                                                                   num_leaves,
                                                                                   feature_fraction,
                                                                                   bagging_fraction,
                                                                                   bagging_freq,
                                                                                   early_stop_rounds)
        save_result(result, path, None)


if __name__ == '__main__':
    # main(offline=False)
    main_leave_one_week(offline=True,
                        mall_ids=["m_6803"],
                        use_hyperopt=False,
                        default_scala=1,
                        use_default_scala=True)  # mall_ids=["m_690", "m_7168", "m_1375", "m_4187", "m_1920", "m_2123"]
