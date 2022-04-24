#!/usr/bin/env python3
import numpy as np
from numpy.linalg import inv
from rogues import randsvd, kms, dorr, lesp # gallery() equivalent for Python

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

# Returns padded bands
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

    return a, b, c


def pad_bands(a, b, c):
    a_padded = np.concatenate(([0], a))
    c_padded = np.concatenate((c, [0]))
    assert(len(b) == len(a_padded))
    assert(len(b) == len(c_padded))
        
    # b is unmodified
    return a_padded, b, c_padded


def read_matrixmarket(filename):
    with open(filename) as f:
        s = f.read()
        ss = s.split('\n')
        l = ss[1].split()
        num_rows = int(l[0])
        num_cols = int(l[1])
        arr = np.zeros((num_rows, num_cols))
        
        for l in ss[2:-1]:
            i, j, x = list(map(float, l.split()))
            arr[int(i) - 1][int(j) - 1] = x

        return arr
    

def read_dense_matrixmarket(filename):
    with open(filename) as f:
        s = f.read()
        ss = s.split('\n')
        l = ss[1].split()
        num_rows = int(l[0])
        # TODO can only read a vector
        #num_cols = int(l[1])
        arr = np.zeros(num_rows)
        
        for i,l in enumerate(ss[2:-1]):
            x = float(l)
            arr[i] = x

        return arr


# TODO: return padded bands
def generate_tridiag(ID, N, unif_low=-1, unif_high=1):
    b_unif = np.random.uniform(unif_low, unif_high, N)
    a_unif = np.random.uniform(unif_low, unif_high, N-1)
    c_unif = np.random.uniform(unif_low, unif_high, N-1)
    
    if ID == 1:
        return pad_bands(a_unif, b_unif, c_unif)

    elif ID == 2:
        b = 1e8*np.ones(N)
        return pad_bands(a_unif, b, c_unif)
    
    elif ID == 3:
        return numpy_matrix_to_bands(np.asarray(lesp(N)))

    elif ID == 4:
        a = a_unif
        mid = int(N//2)
        a[mid], a[mid+1] = 1e-50*a_unif[mid], 1e-50*a_unif[mid+1]

        return pad_bands(a, b_unif, c_unif)

    elif ID == 5:
        # each element of a has 50% chance to be zero
        # without dtype=bool, the indexed array consists of a[0] and a[1]
        mask_a = np.array(np.random.binomial(1, 0.5, N-1), dtype=bool)
        a = np.zeros(N-1)
        a[mask_a] = a_unif[mask_a]

        # each element of c has 50% chance to be zero
        mask_c = np.array(np.random.binomial(1, 0.5, N-1), dtype=bool)
        c = np.zeros(N-1)
        c[mask_c] = c_unif[mask_c]

        return pad_bands(a, b_unif, c)

    elif ID == 6:
        b = 64*np.ones(N)
        return pad_bands(a_unif, b, c_unif)

    elif ID == 7:
        return numpy_matrix_to_bands(inv(kms(N, rho=0.5)))

    elif ID == 8:
        return numpy_matrix_to_bands(
                randsvd(N, kappa=1e15, mode=2, kl=1, ku=1))

    elif ID == 9:
        return numpy_matrix_to_bands(
                randsvd(N, kappa=1e15, mode=3, kl=1, ku=1))

    elif ID == 10:
        return numpy_matrix_to_bands(
                randsvd(N, kappa=1e15, mode=1, kl=1, ku=1))

    elif ID == 11:
        return numpy_matrix_to_bands(
                randsvd(N, kappa=1e15, mode=4, kl=1, ku=1))

    elif ID == 12:
        a = a_unif*1e-50
        return pad_bands(a, b_unif, c_unif)

    elif ID == 13:
        a, b, c = dorr(N, theta=1e-4)
        return pad_bands(a, b, c)

    elif ID == 14:
        b = 1e-8*np.ones(N)
        return pad_bands(a_unif, b, c_unif)

    elif ID == 15:
        b = np.zeros(N)
        return pad_bands(a_unif, b, c_unif)
    
    elif ID == 16:
        return pad_bands(np.ones(N-1), 1e-8*np.ones(N), np.ones(N-1))

    elif ID == 17:
        return pad_bands(np.ones(N-1), 1e8*np.ones(N), np.ones(N-1))

    elif ID == 18:
        return pad_bands(-np.ones(N-1), 4*np.ones(N), -np.ones(N-1))

    elif ID == 19:
        return pad_bands(-np.ones(N-1), 4*np.ones(N), np.ones(N-1))

    elif ID == 20:
        return pad_bands(-np.ones(N-1), 4*np.ones(N), c_unif)


def generate_tridiag_system(mtx_id, N_fine, unif_low, unif_high):
    # Bands of coefficient matrix (with added padding)
    a_fine, b_fine, c_fine = generate_tridiag(mtx_id, N_fine, unif_low, unif_high)

    # Solution
    mtx = bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    x_fine = np.random.normal(3, 1, N_fine)

    # Right-hand side
    d_fine = np.matmul(mtx, x_fine)
    
    return a_fine, b_fine, c_fine, d_fine, x_fine

