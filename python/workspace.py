#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:06:27 2022

@author: archie
"""

# %%
import os

home_dir = os.environ['HOME']
source_dir = '{}/source/repos/arpts'.format(home_dir)
os.chdir('{}/python'.format(source_dir))

# %%
import matrix, partition, rpta
import numpy as np

from scipy.io import mmread

# %%
from main_static import main_static
from main_cond_coarse import main_cond_coarse
from main_random import main_random
from main_rows import main_rows

# %% Linear system
mtx_id = 14
N_fine = 512

A_sp = mmread('{}/mtx/{:02d}-{}'.format(source_dir, mtx_id, N_fine))
A = np.asarray(A_sp.todense())

# %%
a_fine, b_fine, c_fine = matrix.numpy_matrix_to_bands(A)
x_fine = np.random.normal(3, 1, N_fine)
d_fine = np.matmul(A, x_fine)

# %% Verify significance of (relative) size of spikes
def find_spike_ratios(part_begin, part_end, a_fine, b_fine, c_fine):
    spike_ratios = []
    
    for i in range(part_begin, part_end):
        row_max = np.max((abs(a_fine[i]), abs(b_fine[i]), abs(c_fine[i])))
        lower_spike_to_max = abs(c_fine[i]) / row_max
        upper_spike_to_max = abs(a_fine[i]) / row_max
        spike_ratios.append([lower_spike_to_max, upper_spike_to_max])
    
    return spike_ratios

def solve_slice(a_fine, b_fine, c_fine, d_fine, i_begin, i_end):
    x_fine_slice, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
        2, a_fine[i_begin:i_end], b_fine[i_begin:i_end], c_fine[i_begin:i_end], 
        d_fine[i_begin:i_end], [[i_begin, i_end]])
    
    # forward relative error
    fre = np.linalg.norm(x_fine_slice - x_fine[i_begin:i_end]) / np.linalg.norm(x_fine[i_begin:i_end])
    # component-wise relative error
    cre = np.max(np.abs(x_fine_slice - x_fine[i_begin:i_end]) / np.abs(x_fine[i_begin:i_end]))
    
    return x_fine_slice, mtx_coarse, mtx_cond_coarse, fre, cre

# %%
ratios = np.array(find_spike_ratios(15, 64, a_fine, b_fine, c_fine))
M_opt_ratio = np.argmin(ratios, axis=0)[0] + 1 # lower spike ratio (first partition)
M_worst_ratio = np.argsort(ratios, axis=0)[-1][0] + 1

# %%
x_fine_32, mtx_coarse_32, mtx_cond_coarse_32, fre_32, cre_32 = solve_slice(
    a_fine, b_fine, c_fine, d_fine, 0, 32)
x_fine_33, mtx_coarse_33, mtx_cond_coarse_33, fre_33, cre_33 = solve_slice(
    a_fine, b_fine, c_fine, d_fine, 0, 33)
x_fine_opt_ratio, mtx_coarse_opt_ratio, mtx_cond_coarse_opt_ratio, fre_opt_ratio, cre_opt_ratio = solve_slice(
    a_fine, b_fine, c_fine, d_fine, 0, M_opt_ratio)
x_fine_worst_ratio, mtx_coarse_worst_ratio, mtx_cond_coarse_worst_ratio, fre_worst_ratio, cre_worst_ratio = solve_slice(
    a_fine, b_fine, c_fine, d_fine, 0, M_worst_ratio)


# %% Dynamic approach where we start with a fixed partition, then proceed in one direction
# (downwards from upper block A, upwards from lower block A+2), and check how well-conditioned
# the block without the (assumed to be small) spikes is
M = 32
A1 = A[0:2*M, 0:2*M]  # block 0-1
A2 = A[M:3*M, M:3*M]  # block 1-2
A3 = A[0:3*M, 0:3*M]  # block 0-2

a1_fine, b1_fine, c1_fine = matrix.numpy_matrix_to_bands(A1)
d1_fine = d_fine[0:2*M]

a2_fine, b2_fine, c2_fine = matrix.numpy_matrix_to_bands(A2)
d2_fine = d_fine[M:3*M]

a3_fine, b3_fine, c3_fine = matrix.numpy_matrix_to_bands(A3)
d3_fine = d_fine[0:3*M]

# %% eliminate_band() which keeps track of all rows
def eliminate_band(a, b, c, d):
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

        # Scaled partial pivoting
        m_p = max([abs(s_p[0]), abs(s_p[1]), abs(s_p[2])])
        m_c = max([abs(s_c[1]), abs(s_c[2]), abs(s_c[3])])

        if abs(s_c[1])*m_p > abs(s_p[1])*m_c:
            r_c = (-1.0) * s_p[1] / s_c[1]
            r_p = 1.0
        else:
            r_c = 1.0
            r_p = (-1.0) * s_c[1] / s_p[1]

        for k in [0, 2, 3, 4]:
            s_p[k] = r_p * s_p[k] + r_c * s_c[k]

        s_p[1] = s_p[2]
        s_p[2] = s_p[3]
        s_p[3] = 0.0

        yield s_p

def eliminate_band_reversed(a, b, c, d):
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    c_rev = list(reversed(c))
    d_rev = list(reversed(d))
    
    yield from eliminate_band(c_rev, b_rev, a_rev, d_rev)


# %% downwards elimination from block 0
for i, s_p in enumerate(eliminate_band(a1_fine, b1_fine, c1_fine, d1_fine)): # row indices
    # a, b, c, d = s_p
    print(i+1, ":", s_p[0], s_p[1], s_p[2], s_p[3])

# %% upwards elimination from block 2
for i, s_r in enumerate(eliminate_band_reversed(a2_fine, b2_fine, c2_fine, d2_fine)):
    # c, b, a, d = s_r
    print(3*M-(i+2), ":", s_r[2], s_r[1], s_r[0], s_r[3])
    
# %% check 2x2 systems (ignoring values of lower and upper spike)
mtx_conds, mtx_dets = [np.Inf], [0]
gen_p = eliminate_band(a1_fine, b1_fine, c1_fine, d1_fine)
gen_r = eliminate_band_reversed(a2_fine, b2_fine, c2_fine, d2_fine)

for i in range(1, 2*M):  # row indices
    s_p = next(gen_p)
    s_r = next(gen_r)
    
    mtx_check = np.array([[s_p[1], s_p[2]], [s_r[2], s_r[1]]])
    # mtx_check = np.array([[s_p[0], s_p[1]], [s_r[1], s_r[0]]])
    mtx_conds.append(np.linalg.cond(mtx_check))
    mtx_dets.append(np.abs(np.linalg.det(mtx_check)))

# %%
x_fine_conds, mtx_coarse_conds, mtx_cond_coarse_conds = rpta.reduce_and_solve(
    4, a3_fine, b3_fine, c3_fine, d3_fine, [[0, 42], [42, 96]])

fre_conds = np.linalg.norm(
    x_fine_conds - x_fine[0:96]) / np.linalg.norm(x_fine[0:96])