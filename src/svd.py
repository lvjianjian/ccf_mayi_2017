#!/usr/bin/env python
# encoding=utf-8

"""
@Author: zhongjianlv

@Create Date: 17-10-23, 17:18

@Description:

@Update Date: 17-10-23, 17:18
"""

from util import *
import scipy.sparse as sp
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error


def main(mall_id):
    df, train_cache, test_cache = get_wifi_cache(mall_id, default=0)
    train_matrix = train_cache[2]
    test_matrix = test_cache[2]

    print "svd"
    u, s, vt = svds(train_matrix, k=50)
    s_diag_matrix = np.diag(s)
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)

    print train_matrix
    print 'User-based CF MSE: ' + str(np.sqrt(mean_squared_error(X_pred, train_matrix)))
    print X_pred


if __name__ == '__main__':
    main("m_7168")
