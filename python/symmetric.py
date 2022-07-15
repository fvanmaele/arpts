#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:11:15 2022

@author: archie
"""

import matrix
import numpy as np

# %%
def eliminate_band_iter(a, b, c, d, pivoting):
    M = len(a)
    assert(M > 1) # band should at least have one element

    # to save a, b, c, d, spike
    s_p = [0.0] * 5
    s_c = [0.0] * 5

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
    s_new = [0.0] * (M - 1)  # spike
    b_new = [0.0] * (M - 1)
    c_new = [0.0] * (M - 1)
    d_new = [0.0] * (M - 1)
    
    for j, s_p in enumerate(eliminate_band_iter(a, b, c, d, pivoting)):
        s_new[j] = s_p[0]
        b_new[j] = s_p[1]
        c_new[j] = s_p[2]
        d_new[j] = s_p[4]

    return s_new, b_new, c_new, d_new


# %%
def rpta_symmetric(a_fine, b_fine, c_fine, d_fine, partition, pivoting='scaled_partial'):
    assert(len(a_fine) > 1) # band should at least have one element
    N_coarse = len(partition)*2
    a_coarse = [0.0] * N_coarse
    b_coarse = [0.0] * N_coarse
    c_coarse = [0.0] * N_coarse
    d_coarse = [0.0] * N_coarse

    # TODO: better way of saving vectors for each block
    saved_arrays = [[None, None]]*len(partition)

    for part_id, part_bounds in enumerate(partition):
        part_begin, part_end = part_bounds
        
        s_lower, b_lower, c_lower, d_lower = eliminate_band_expand(
            a_fine[part_begin:part_end],
            b_fine[part_begin:part_end],
            c_fine[part_begin:part_end],
            d_fine[part_begin:part_end], pivoting)
        saved_arrays[part_id][0] = s_lower, b_lower, c_lower, d_lower

        a_upper, b_upper, s_upper, d_upper = eliminate_band_expand(
            list(reversed(c_fine[part_begin:part_end])),
            list(reversed(b_fine[part_begin:part_end])),
            list(reversed(a_fine[part_begin:part_end])),
            list(reversed(d_fine[part_begin:part_end])), pivoting)
        saved_arrays[part_id][1] = a_upper, b_upper, s_upper, d_upper

        a_coarse[2 * part_id] = s_upper[-1]
        b_coarse[2 * part_id] = b_upper[-1]
        c_coarse[2 * part_id] = a_upper[-1]
        d_coarse[2 * part_id] = d_upper[-1]

        a_coarse[2 * part_id + 1] = s_lower[-1]
        b_coarse[2 * part_id + 1] = b_lower[-1]
        c_coarse[2 * part_id + 1] = c_lower[-1]
        d_coarse[2 * part_id + 1] = d_lower[-1]

    # TODO: support recursion
    # Solve coarse system (interface degrees of freedom)
    mtx_coarse = matrix.bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)
    x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
    
    # Solve blocks 
    for part_id, part_bounds in enumerate(partition):
        part_begin, part_end = part_bounds
        x1 = x_coarse[part_id]
        xn = x_coarse[part_id+1]
        
        
    # TODO: solve 2x2 linear systems
    return a_coarse, b_coarse, c_coarse, d_coarse