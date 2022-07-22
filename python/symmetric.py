#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:11:15 2022

@author: archie
"""

import matrix
import numpy as np
import sys

# %%
def eliminate_band_iter(a, b, c, d, pivoting):
    M = len(a)
    assert(M > 1) # band should at least have one element

    # to save a, b, c, d, spike
    s_p = [0.0] * 5
    s_c = [0.0] * 5

    # return first two noop iterations so that the result has length M
    yield a[0], b[0], c[0], 0.0, d[0]

    s_p[0] = a[1]
    s_p[1] = b[1]
    s_p[2] = c[1]
    s_p[3] = 0.0
    s_p[4] = d[1]

    yield s_p

    for j in range(2, M):
        s_c[0] = 0.0
        s_c[1] = a[j]
        s_c[2] = b[j]
        s_c[3] = c[j]
        s_c[4] = d[j]

        if pivoting == "scaled_partial":
            m_p = max([abs(s_p[0]), abs(s_p[1]), abs(s_p[2])])
            m_c = max([abs(s_c[1]), abs(s_c[2]), abs(s_c[3])])
        elif pivoting == "partial":
            m_p = 1.0
            m_c = 1.0
        elif pivoting == "none":
            m_p = 0.0
            m_c = 0.0

        if abs(s_c[1])*m_p > abs(s_p[1])*m_c:
            # print("{} * {} / {} (pivoted)".format(-1, s_p[1], s_c[1]))
            r_c = (-1.0) * s_p[1] / s_c[1]
            r_p = 1.0
        else:
            # print("{} * {} / {}".format(-1, s_c[1], s_p[1]))
            r_c = 1.0
            r_p = (-1.0) * s_c[1] / s_p[1]

        for k in [0, 2, 3, 4]:
            # print("{} * {} + {} * {}".format(r_p, s_p[k], r_c, s_c[k]))
            s_p[k] = r_p * s_p[k] + r_c * s_c[k]

        s_p[1] = s_p[2]
        s_p[2] = s_p[3]
        s_p[3] = 0.0

        yield s_p

def eliminate_band_expand(a, b, c, d, pivoting):
    M = len(a)
    s_new = [0.0] * M  # spike
    b_new = [0.0] * M
    c_new = [0.0] * M
    d_new = [0.0] * M

    for j, s_p in enumerate(eliminate_band_iter(a, b, c, d, pivoting)):
        s_new[j] = s_p[0]
        b_new[j] = s_p[1]
        c_new[j] = s_p[2]
        d_new[j] = s_p[4]

    return s_new, b_new, c_new, d_new

def eliminate_band(a, b, c, d, pivoting):
    for v in eliminate_band_iter(a, b, c, d, pivoting):
        _ = v

    return v[0], v[1], v[2], v[4]

# %%
# TODO: this only supports a single recursion step
def rpta_symmetric(a_fine, b_fine, c_fine, d_fine, partition, pivoting='scaled_partial'):
    assert(len(a_fine) > 1) # band should at least have one element
    N_coarse = len(partition)*2
    a_coarse = [0.0] * N_coarse
    b_coarse = [0.0] * N_coarse
    c_coarse = [0.0] * N_coarse
    d_coarse = [0.0] * N_coarse

    for part_id, part_bounds in enumerate(partition):
        part_begin, part_end = part_bounds
        
        # downwards elimination
        s_lower, b_lower, c_lower, d_lower = eliminate_band(
            a_fine[part_begin:part_end],
            b_fine[part_begin:part_end],
            c_fine[part_begin:part_end],
            d_fine[part_begin:part_end], pivoting)

        a_coarse[2 * part_id + 1] = s_lower
        b_coarse[2 * part_id + 1] = b_lower
        c_coarse[2 * part_id + 1] = c_lower
        d_coarse[2 * part_id + 1] = d_lower

        # upwards elimination
        s_upper, b_upper, a_upper, d_upper = eliminate_band(
            list(reversed(c_fine[part_begin:part_end])),
            list(reversed(b_fine[part_begin:part_end])),
            list(reversed(a_fine[part_begin:part_end])),
            list(reversed(d_fine[part_begin:part_end])), pivoting)

        a_coarse[2 * part_id] = a_upper
        b_coarse[2 * part_id] = b_upper
        c_coarse[2 * part_id] = s_upper
        d_coarse[2 * part_id] = d_upper

    # Solve coarse system (reduction step, solve interface between blocks)
    mtx_coarse = matrix.bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)
    x_coarse = np.linalg.solve(mtx_coarse, d_coarse)

    x_fine = [] # solution of fine system
    for part_id, part_bounds in enumerate(partition):
        part_begin, part_end = part_bounds
        
        x0 = x_coarse[2*part_id]   # x_fine[part_begin]
        xm = x_coarse[2*part_id+1] # x_fine[part_end]
        x_fine.append(x0)
        # print("{} {:>20.6e}".format(part_end, x0))

        # XXX: compute everything again, because I can't find an immediate way to make copies of the
        # arrays computed above - e.g. neither deepcopy() or copy() have any effect, and only the
        # arrays for the last partition are stored.
        # XXX: if we are not storing the vectors anyway, we can iterate over the elimination
        # and compute x_j on-demand
        
        # downwards elimination
        s_lower, b_lower, c_lower, d_lower = eliminate_band_expand(
            a_fine[part_begin:part_end],
            b_fine[part_begin:part_end],
            c_fine[part_begin:part_end],
            d_fine[part_begin:part_end], pivoting)
        
        # upwards elimination
        s_upper, b_upper, a_upper, d_upper = eliminate_band_expand(
            list(reversed(c_fine[part_begin:part_end])),
            list(reversed(b_fine[part_begin:part_end])),
            list(reversed(a_fine[part_begin:part_end])),
            list(reversed(d_fine[part_begin:part_end])), pivoting)

        # store upwards elimination in same order as downwards elimination
        a_upper_rev = list(reversed(a_upper))
        b_upper_rev = list(reversed(b_upper))
        s_upper_rev = list(reversed(s_upper))
        d_upper_rev = list(reversed(d_upper))

        xjpp_prev = None
        xjpp = None
        det_j_prev = None
        det_j = None

        for idx, j in enumerate(range(part_begin+1, part_end-1), start=2):
            if j == part_end-2:  # iteration end
                # XXX: in this case we can either take the solution of the previous step, or solve
                # x_{M-1} directly from x_{M}
                assert(xjpp_prev is not None)
                x_fine.append(xjpp_prev)

            else:
                mtx_j = np.array([[b_lower[idx-1], c_lower[idx-1]], 
                                  [a_upper_rev[idx], b_upper_rev[idx]]])
                rhs_j = np.array([d_lower[idx-1] - x0*s_lower[idx-1],
                                  d_upper_rev[idx] - xm*s_upper_rev[idx]])
                # invert 2x2 system
                det_j = mtx_j[0][0]*mtx_j[1][1] - mtx_j[0][1]*mtx_j[1][0]
                adj_j = np.array([[ mtx_j[1][1], -mtx_j[0][1]],
                                  [-mtx_j[1][0],  mtx_j[0][0]]])
                inv_j = 1/det_j * adj_j

                # retrieve solutions
                xj, xjpp = inv_j @ rhs_j

                if xjpp_prev is not None:
                    # Decide which solution to choose based on condition of linear system
                    if abs(det_j_prev) < abs(det_j):
                        x_fine.append(xj)
                    else:
                        x_fine.append(xjpp_prev)
    
                    # ratio = abs(det_j_prev) / abs(det_j)
                    # if ratio > 10:
                    #     print("diag: det changed with at least 1 order of magnitude", file=sys.stderr)
                    #     print("{} (x = {}) vs. {} (x = {})".format(
                    #         det_j_prev, xjpp_prev, det_j, xj), file=stderr)
                else:
                    x_fine.append(xj)
    
                # assign solution/det for next iteration
                xjpp_prev = xjpp
                det_j_prev = det_j

            # xj, xjpp = np.linalg.solve(mtx_j, rhs_j)
            print("{} {:>20.6e} {:>20.6e} {:>20.6e} {:>20.6e}".format(
                j, xj, xjpp, det_j, np.linalg.cond(mtx_j)))

        # print("{} {:>20.6e}".format(part_end, xm))
        x_fine.append(xm)
    
    return x_fine


# %% Test cases
if __name__ == "__main__" or not hasattr(sys, 'ps1'):
    # Test for single partition
    B = np.array([[1, 2, 0, 0],
                  [2, 1, 2, 0],
                  [0, 2, 1, 2],
                  [0, 0, 2, 1]])
    Ba, Bb, Bc = matrix.numpy_matrix_to_bands(B)
    Bd = np.array([1]*4)

    Bsol = rpta_symmetric(Ba, Bb, Bc, Bd, [[0,4]], 'partial')
    assert(Bsol == [-1., 1., 1., -1.])

    # Test for multiple partitions
    B2 = np.vstack((np.hstack((B, np.zeros((4, 4)))), np.hstack((np.zeros((4,4)), B))))
    B2a, B2b, B2c = matrix.numpy_matrix_to_bands(B2)
    B2d = np.array([1]*8)

    B2sol = rpta_symmetric(B2a, B2d, B2c, B2d, [[0,4], [4,8]], 'partial')
    assert(B2sol == [-1., 1., 1., -1., -1., 1., 1., -1.])