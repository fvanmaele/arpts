#!/usr/bin/env python3
import numpy as np
from scipy import sparse
from numpy.linalg import inv
from rogues import randsvd, kms, dorr, lesp # gallery() equivalent for Python

def generate_matrix(Id, N, unif_low=-1, unif_high=1):
    b_unif = np.random.uniform(unif_low, unif_high, N)
    a_unif = np.random.uniform(unif_low, unif_high, N-1)
    c_unif = np.random.uniform(unif_low, unif_high, N-1)

    # https://docs.python.org/3.10/whatsnew/3.10.html#pep-634-structural-pattern-matching
    match ID:
        case 1:
            return tridiag(a_unif, b_unif, c_unif)
        
        case 2:
            b = 1e8*np.ones(N)
            return tridiag(a_unif, b, c_unif)
        
        case 3:
            return lesp(N)

        case 4:
            a = a_unif
            a[np.floor(N/2)] = 1e-50 * a_unif[np.floor(N/2)]
            a[np.floor(N/2)+1] = 1e-50 * a_unif[np.floor(N/2)+1]

            return tridiag(a, b_unif, c_unif)

        case 5:
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

        case 6:
            b = 64*np.ones(N)
            return tridiag(a_unif, b, c_unif)

        case 7:
            return inv(kms(N, rho=0.5))

        case 8:
            return randsvd(N, kappa=1e15, mode=2, kl=1, ku=1)

        case 9:
            return randsvd(N, kappa=1e15, mode=3, kl=1, ku=1)

        case 10:
            return randsvd(N, kappa=1e15, mode=1, kl=1, ku=1)

        case 11:
            return randsvd(N, kappa=1e15, mode=4, kl=1, ku=1)

        case 12:
            a = a_unif*1e-50
            return tridiag(a, b_unif, c_unif)

        case 13:
            return dorr(N, theta=1e-4)

        case 14:
            b = 1e-8*np.ones(N)
            return tridiag(a_unif, b, c_unif)

        case 15:
            b = np.zeros(N, 1)
            return tridiag(a_unif, b, c_unif)
        
        case 16:
            return tridiag(np.ones(N-1), 1e-8*np.ones(N), np.ones(N-1))

        case 17:
            return tridiag(np.ones(N-1), 1e8*np.ones(N), np.ones(N-1))

        case 18:
            return tridiag(-np.ones(N-1), 4*np.ones(N), -np.ones(N-1))

        case 19:
            return tridiag(-np.ones(N-1), 4*np.ones(N), np.ones(N-1))

        case 20:
            return tridiag(-np.ones(N-1), 4*np.ones(N), c_unif)


def tridiag(a, b, c, sparse=0):
    # Default to dense matrix
    sparse_result = 0

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
