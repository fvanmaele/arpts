#!/usr/bin/env python3
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from rogues import randsvd, kms, dorr, lesp # gallery() equivalent for Python


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


def tridiag(a, b, c, sparse_result=0):
    # Precondition checks
    N = len(b)
    assert(len(a) == N-1)
    assert(len(c) == N-1)

    if sparse_result == 1:
        diagonals = [a, b, c]
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.diags.html
        return sparse.diags(diagonals, [-1, 0, 1], shape=(N, N))
    else:
        return np.diag(a, -1) + np.diag(b) + np.diag(c, 1)


def generate_matrix(ID, N, unif_low=-1, unif_high=1):
    b_unif = np.random.uniform(unif_low, unif_high, N)
    a_unif = np.random.uniform(unif_low, unif_high, N-1)
    c_unif = np.random.uniform(unif_low, unif_high, N-1)

    if ID == 1:
        return tridiag(a_unif, b_unif, c_unif)

    elif ID == 2:
        b = 1e8*np.ones(N)
        return tridiag(a_unif, b, c_unif)
    
    elif ID == 3:
        return np.asarray(lesp(N))

    elif ID == 4:
        a = a_unif
        mid = int(N//2)
        a[mid] = 1e-50 * a_unif[mid]
        a[mid+1] = 1e-50 * a_unif[mid+1]

        return tridiag(a, b_unif, c_unif)

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

        return tridiag(a, b_unif, c)

    elif ID == 6:
        b = 64*np.ones(N)
        return tridiag(a_unif, b, c_unif)

    elif ID == 7:
        return inv(kms(N, rho=0.5))

    elif ID == 8:
        return randsvd(N, kappa=1e15, mode=2, kl=1, ku=1)

    elif ID == 9:
        return randsvd(N, kappa=1e15, mode=3, kl=1, ku=1)

    elif ID == 10:
        return randsvd(N, kappa=1e15, mode=1, kl=1, ku=1)

    elif ID == 11:
        return randsvd(N, kappa=1e15, mode=4, kl=1, ku=1)

    elif ID == 12:
        a = a_unif*1e-50
        return tridiag(a, b_unif, c_unif)

    elif ID == 13:
        a, b, c = dorr(N, theta=1e-4)
        return tridiag(a, b, c)

    elif ID == 14:
        b = 1e-8*np.ones(N)
        return tridiag(a_unif, b, c_unif)

    elif ID == 15:
        b = np.zeros(N)
        return tridiag(a_unif, b, c_unif)
    
    elif ID == 16:
        return tridiag(np.ones(N-1), 1e-8*np.ones(N), np.ones(N-1))

    elif ID == 17:
        return tridiag(np.ones(N-1), 1e8*np.ones(N), np.ones(N-1))

    elif ID == 18:
        return tridiag(-np.ones(N-1), 4*np.ones(N), -np.ones(N-1))

    elif ID == 19:
        return tridiag(-np.ones(N-1), 4*np.ones(N), np.ones(N-1))

    elif ID == 20:
        return tridiag(-np.ones(N-1), 4*np.ones(N), c_unif)


def generate_linear_system(mtx_id, N_fine, unif_low, unif_high):
    # Solution
    x_fine = np.random.normal(3, 1, N_fine)
    
    # Coefficient matrix
    mtx = generate_matrix(mtx_id, N_fine, unif_low, unif_high)
    a_fine, b_fine, c_fine = numpy_matrix_to_bands(mtx)
    
    # Right-hand side
    d_fine = np.matmul(mtx, x_fine)
    
    return a_fine, b_fine, c_fine, d_fine, x_fine


def main():
    N = 512
    for ID in range(1, 21):
        print('generating matrix {} of size {}'.format(ID, N))
        mtx_n, mtx_m = np.shape(generate_matrix(ID, N))
        assert(mtx_n == N)
        assert(mtx_m == N)
        
if __name__ == "__main__":
    main()