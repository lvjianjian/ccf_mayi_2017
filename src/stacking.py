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
                if _cls[0].startswith("lgb"):
                    for _i in range(self.kfold):
                        lgb_name = _cls[0] + "_" + str(_i)
                        self.spetial_base_clses_params[lgb_name] = (lgb_name, _cls[1].copy())
                elif _cls[0].startswith("xgb"):
                    for _i in range(self.kfold):
                        xgb_name = _cls[0] + "_" + str(_i)
                        self.spetial_base_clses_params[xgb_name] = (xgb_name, _cls[1].copy())
                else:
                    base_name = _cls[0]
                    for _i in range(self.kfold):
                        self.sklearn_models[base_name + "_{}".format(_i)] = clone(_cls[1])


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
            p = dict(zip(clss, p.transpose()))
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
                if _cls[0].startswith("lgb"):
                    p = self._lgb_train(_cls, train_x, train_y, valid_x, valid_y)
                    p_rs.append(p)
                    if valid_x is not None:
                        p = self._lgb_predict(_cls[0], valid_x, self.use_prob)
                        v_rs.append(p)
                else:
                    _cls[1].fit(train_x, train_y)
                    p = self._sklearn_predict(_cls[1], train_x, self.use_prob)
                    p_rs.append(p)
                    if valid_x is not None:
                        p = self._sklearn_predict(_cls[1], valid_x, self.use_prob)
                        v_rs.append(p)

        else:
            for _cls in self.base_clses:
                p_part = []
                y_index = []
                v_part = []
                if _cls[0].startswith("lgb"):  # lgb or xgb
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
                        name = _cls[0]
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
        if self.meta_cls[0].startswith("lgb"):
            self._lgb_train(self.meta_cls, new_train, train_y, valid_x, valid_y)
        else:
            self.meta_cls[1].fit(new_train, train_y)

    def predict(self, test_x):
        p_rs = []
        if not self.cv:
            for _cls in self.base_clses:
                if _cls[0].startswith("lgb"):
                    p = self._lgb_predict(_cls[0], test_x, self.use_prob)
                    p_rs.append(p)
                else:
                    p = self._sklearn_predict(_cls[1], test_x, self.use_prob)
                    p_rs.append(p)

        else:
            for _cls in self.base_clses:
                p_part = []
                if _cls[0].startswith("lgb"):  # lgb or xgb
                    for _i in range(self.kfold):
                        lgb_name = _cls[0] + "_" + str(_i)
                        p_part.append(self._lgb_predict(lgb_name, test_x, self.use_prob))

                else:  # sklearn
                    for _i in range(self.kfold):
                        name = _cls[0]
                        model = self.sklearn_models[name + "_" + str(_i)]
                        p = self._sklearn_predict(model, test_x, self.use_prob)
                        p_part.append(p)
                p_rs.append(np.mean(np.stack(p_part, axis=2), axis=2))

        new_train = np.hstack(p_rs)
        # 次学习器
        if self.meta_cls[0].startswith("lgb"):
            p = self._lgb_predict(self.meta_cls[0], new_train)
        else:
            p = self._sklearn_predict(self.meta_cls[1], new_train)
        return p.reshape((-1,))


