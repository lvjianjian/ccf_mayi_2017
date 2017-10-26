# encoding = utf-8
import lightgbm as lgb
from features_extractors import *
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from common import *


class predictors(object):
    def __init__(self, train_features, train_labels,
                 test_features, config, label_encoder):
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.best_iteration = 0
        self.label_encoder = label_encoder

        # register the predictor
        cmd = "self.predictor = self." + config["name"] + "_predict"
        exec (cmd)

    def kfold_split(self, mall_id, params, train_matrix, y,data):
        print mall_id, params
        return KFold(n_splits=params["kfold"]).split(train_matrix, y)

    def leave_one_week_split(self, mall_id, params, train_matrix, y,data):
        return [common.get_last_one_week_index(data)]


    def lgb_predict(self, mall_id, params, data):
        # predict the results and write in the f
        # TODO
        offline = params["offline"]

        train_matrix = self.train_features
        train_labels = self.train_labels
        test_matrix = self.test_features
        y = self.train_labels
        offline_predicts = []
        offline_reals = []

        cmd = "self.split = self." + params["train_valid_split_method"] + \
                            "_split(mall_id,params,train_matrix,train_labels,data)"
        print cmd
        exec (cmd)
        best_interations = []
        _index = 0
        best_iterations = []
        for _train_index, _valid_index in self.split:
            _train_x = train_matrix[_train_index]
            _train_y = y[_train_index]
            # _train = lgb.Dataset(_train_x,
            #                     label=_train_y)
            _train = [_train_x, _train_y]
            _valid_x = train_matrix[_valid_index]
            _valid_y = y[_valid_index]
            _valid = [_valid_x, _valid_y]

            bst = self.lgb_model_train(params, _train, _valid)

            predict = np.argmax(bst.predict(_valid_x,
                                            num_iteration=bst.best_iteration),
                                axis=1).astype(int)

            predict = self.label_encoder.inverse_transform(predict)
            offline_predicts.append(predict)
            offline_reals.append(self.label_encoder.inverse_transform(_valid_y))
            _index += 1
            best_iterations.append(bst.best_iteration)
        best_iteration = int(np.mean(best_iterations))
        self.best_iteration = best_iteration

        if not offline:
            bst = self.lgb_model_train(params,
                                       [train_matrix, train_labels],
                                       offline=False
                                       )
            predict = np.argmax(bst.predict(test_matrix, best_iteration),
                                axis=1).astype(int)
            predict = self.label_encoder.inverse_transform(predict)
            return offline_predicts, offline_reals, \
                   predict, test_all[np.in1d(test_all.index,
                                             test_index)].row_id.values
        return offline_predicts, offline_reals


    def lgb_model_train(self, params, train_set, valid_set=None, offline=True):
        [_train_x, _train_y] = train_set
        [_valid_x, _valid_y] = valid_set

        if offline:
            _train = lgb.Dataset(_train_x,
                                 label=_train_y)
            _valid = lgb.Dataset(_valid_x,
                                 label=_valid_y,
                                 reference=_train)
            bst = lgb.train(params["hyperparams"],
                            _train,
                            params["n_round"],
                            valid_sets=_valid,
                            early_stopping_rounds=params["early_stop_rounds"])
        else:
            train = lgb.train(_train_x, _train_y)
            bst = lgb.train(params["hyperparams"],
                            _train,
                            self.best_iteration)
        return bst
# class predict_model_1(object):.....
