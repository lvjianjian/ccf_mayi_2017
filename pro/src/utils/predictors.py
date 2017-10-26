# encoding = utf-8
import lightgbm as lgb
from features_extractors import *
from sklearn.model_selection import train_test_split, KFold

class predictors(object):
	def __init__(self, train_features, train_labels,
                    test_features, config):
        
        self.train_features = train_features
		self.train_labels = train_labels
        self.test_features = test_features
        self.best_iteration = 0
        self.kfold = config["kfold"]
        self.offline_predicts = []
        self.offline_reals = []
        self.all_rowid = []
        self.all_predict = []
        
        for _ in range(self.kfold):
            offline_predicts.append({})
            offline_reals.append({})

        # register the predictor
        cmd = "self.predictor = "+ config["model"] + "_predict"
        
        exec(cmd)

	def lgb_predict(self, parameters):
		# predict the results and write in the f
		# TODO
        
        kfold = parameters["kfold"]
        offline = parameters["offline"]
        model_param = parameters["model_param"]
        train_matrix = self.train_features
        train_labels = self.train_labels
        test_matrix = self.test_features
        y = self.train_labels

        if offline:
            train_matrix, test_matrix, y, test_y = train_test_split(train_matrix, 
                                                                    train_labels, 
                                                                    test_size=0.1)
        
        train_x, valid_x, train_y, valid_y = train_test_split(train_matrix,
                                                                y
                                                                )
        
        result = {}
        kf = KFold(n_split=kfold)
        best_interations = []
        _index = 0
        best_iterations = []
        for _train_index, _valid_index in kf.split(train_matrix):
            _train_x = train_matrix[_train_index]
            _train_y = y[_train_index]
            _train = lgb.Dataset(_train_x, 
                                label=_train_y)

            _valid_x = train_matrix[_valid_index]
            _valid_y = y[_valid_index]
            _valid = lgb.Dataset(_valid_x, 
                                label=_valid_y, 
                                reference=_train)

            bst = lgb_model_train(model_param, _train, _valid)

            predict = np.argmax(bst.predict(_valid_x, 
                                num_iteration=bst.best_iteration), 
                                axis=1).astype(int)

            predict = label_encoder.inverse_transform(predict)
            offline_predicts[_index][mall_id] = predict
            offline_reals[_index][mall_id] = label_encoder.inverse_transform(_valid_y)
            _index += 1
            best_iterations.append(bst.best_iteration)
        best_iteration = int(np.mean(best_iterations))
        self.best_iteration = best_iteration
        
        if not offline:  # 线上
            bst = lgb_model_train(  params, 
                                    [train_matrix, train_labels], 
                                    offline=False
                                    )
            predict = np.argmax(bst.predict(test_matrix, best_iteration), 
                                axis=1).astype(int)
            predict = label_encoder.inverse_transform(predict)
            all_predict[mall_id] = predict
            all_rowid[mall_id] = test_all[np.in1d(test_all.index,
                                         test_index)].row_id.values

    def lgb_model_train(self, params, [_train_x, _train_y], [_valid_x, _valid_y]=None, 
                        offline=True):
        if offline:
            _train = lgb.Dataset(_train_x, 
                                label=_train_y)
            _valid = lgb.Dataset(_valid_x, 
                                label=_valid_y, 
                                reference=_train)
            bst = lgb.train(params["hyperparam"],
                            _train,
                            params["n_round"],
                            valid_sets=_valid,
                            early_stopping_rounds=params["early_stop_rounds"])
        else:
            train = lgb.train(_train_x, _train_y)
            bst = lgb.train(params["hyperparam"],
                            _train,
                            self.best_iteration)
        return bst

# class predict_model_1(object):.....
