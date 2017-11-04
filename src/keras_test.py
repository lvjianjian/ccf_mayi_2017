#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-11-3, 13:21

@Description:

@Update Date: 17-11-3, 13:21
"""

from util import *
from keras.layers import Input, Dense, Activation,Dropout
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
import os


def main(mall_id, part_shops, use_cache_model):
    if not os.path.exists("../result/keras_model"):
        os.mkdir("../result/keras_model")

    train_all = load_train()
    test_all = load_testA()
    shop_info = load_shop_info()
    train = train_all[train_all.mall_id == mall_id]
    test = test_all[test_all.mall_id == mall_id]
    y = train.shop_id.values
    le = LabelEncoder().fit(y)
    y = le.transform(y)
    num_class = len(np.unique(y))
    df, train_cache, test_cache = get_wifi_cache2(mall_id)
    train_matrix = train_cache[2]
    test_matrix = test_cache[2]

    train_matrix_origin_all = train_matrix.copy()
    test_matrix_origin_all = test_matrix.copy()

    choose_strong_wifi_index = choose_string_wifi_index(-90, 6, train_matrix)

    train_strong_matrix = train_matrix[:, choose_strong_wifi_index]
    test_strong_matrix = test_matrix[:, choose_strong_wifi_index]

    input_f_n = train_strong_matrix.shape[1]
    print "strong matrix shape", input_f_n

    _train_index, _valid_index = get_last_one_week_index(train)

    _train_x = train_strong_matrix[_train_index]
    _train_y = y[_train_index]
    _valid_x = train_strong_matrix[_valid_index]
    _valid_y = y[_valid_index]

    mms = MinMaxScaler().fit(_train_x)
    _train_mms_x = mms.transform(_train_x)

    _train_categorical_y = np_utils.to_categorical(_train_y, num_class)

    _valid_mms_x = mms.transform(_valid_x)

    _valid_categorical_y = np_utils.to_categorical(_valid_y, num_class)

    # keras model
    inp = Input(shape=(input_f_n,))
    out = Dense(num_class * 5, activation="relu")(inp)
    out = Dense(num_class * 3, activation="relu")(out)
    out = Dropout(0.3)(out)
    out = Dense(num_class, activation="relu")(out)
    out = Dropout(0.1)(out)
    out = Dense(num_class, activation="softmax")(out)
    early_stop = EarlyStopping(monitor="val_categorical_accuracy",patience=5)
    model = Model(inputs=inp, outputs=out)
    sgd = SGD(lr=0.003)
    model.compile(sgd, "categorical_crossentropy", metrics=["categorical_accuracy"])
    print model.summary()
    if use_cache_model and os.path.exists("../result/keras_model/{}.h5".format(mall_id)):
        model.load_weights("../result/keras_model/{}.h5".format(mall_id), by_name=True)

    model.fit(_train_mms_x,
              _train_categorical_y,
              batch_size=1,
              epochs=200,
              validation_data=(_valid_mms_x, _valid_categorical_y)

              )
    if use_cache_model:
        model.save_weights("../result/keras_model/{}.h5".format(mall_id))


if __name__ == '__main__':
    main("m_7168", None, True)
