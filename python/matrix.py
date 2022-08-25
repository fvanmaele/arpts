#!/usr/bin/env python3
import numpy as np
from scipy.sparse import coo_matrix


def bands_to_numpy_matrix(a, b, c):
    N = len(a)
    mtx = np.zeros((N, N))
    
    for row_id in range(N):
        mtx[row_id][row_id] = b[row_id]
        if row_id > 0:
            mtx[row_id][row_id - 1] = a[row_id]
        if row_id < N - 1:
            mtx[row_id][row_id + 1] = c[row_id]

    return mtx


def bands_tridiag(a, b, c):
    # b is unmodified
    return np.concatenate(([0], a)), b, np.concatenate((c, [0]))


def bands_mv(a, b, c, x):
    Ax = np.zeros(len(a))
    n = len(a)

    Ax[0] = b[0]*x[0] + c[0]*x[1]
    for i in range(1, n-1):
        Ax[i] = a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1]
    
    Ax[n-1] = a[n-1]*x[n-2] + b[n-1]*x[n-1]
    return Ax


def numpy_matrix_to_bands(mtx):
    N = mtx.shape[0]
    a = [0.0] * N
    b = [0.0] * N
    c = [0.0] * N
    
    for row_id in range(N):
        b[row_id] = mtx[row_id][row_id]
        if row_id > 0:
            a[row_id] = mtx[row_id][row_id - 1]
        if row_id < N - 1:
            c[row_id] = mtx[row_id][row_id + 1]

    return a, b, c  # padded bands


def scipy_matrix_to_bands(mtx):
    if isinstance(mtx, coo_matrix):
        a_fine = mtx.diagonal(k=-1)
        b_fine = mtx.diagonal()
        c_fine = mtx.diagonal(k=1)
        return bands_tridiag(a_fine, b_fine, c_fine)
    
    else:
        return numpy_matrix_to_bands(mtx)
