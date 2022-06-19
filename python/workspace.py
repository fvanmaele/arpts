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


# %% compute lower spike ratio (first partition, offset = 0)
ratios = np.array(find_spike_ratios(15, 64, a_fine, b_fine, c_fine))
M_opt_ratio = np.argmin(ratios, axis=0)[0] + 1
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


# %% eliminate_band() which keeps track of all rows
def eliminate_band(a, b, c, d, pivoting='scaled_partial'):
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

def eliminate_band_reversed(a, b, c, d):
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    c_rev = list(reversed(c))
    d_rev = list(reversed(d))
    
    yield from eliminate_band(c_rev, b_rev, a_rev, d_rev)

def print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    # M = len(range(begin, end))
    a = a_fine[begin:end]
    b = b_fine[begin:end]
    c = c_fine[begin:end]
    d = d_fine[begin:end]

    for i, s_p in enumerate(eliminate_band(a, b, c, d, pivoting)):
        # a, b, c, d = s_p
        print(begin+i+1, ":", s_p[0], s_p[1], s_p[2], s_p[3], s_p[4])

def print_upwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    M = len(range(begin, end))
    a_rev = list(reversed(a_fine[begin:end]))
    b_rev = list(reversed(b_fine[begin:end]))
    c_rev = list(reversed(c_fine[begin:end]))
    d_rev = list(reversed(d_fine[begin:end]))

    for i, s_r in enumerate(eliminate_band(c_rev, b_rev, a_rev, d_rev, pivoting)):
        # c, b, a, d = s_r
        print(M-(i+1), ":", s_r[2], s_r[1], s_r[0], s_r[3], s_r[4])


# %% downwards elimination (block 0, M = 32, 33)
print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, 16, 64, 'scaled_partial')

# %% upwards elimination (block 0, M = 32)
print_upwards_elimination(a_fine, b_fine, c_fine, d_fine, 0, 32, 'scaled_partial')

# %%
def split_block_reduce_inner_vals(a_fine, b_fine, c_fine, d_fine, begin, end):
    mtx_conds, mtx_dets = [np.Inf], [0]
    a = a_fine[begin:end]
    b = b_fine[begin:end]
    c = c_fine[begin:end]
    d = d_fine[begin:end]
    
    gen_p = eliminate_band(a, b, c, d)

    for i in range(begin+1, end-1):  # row indices, a[1] ... a[M-1]
        # print("i: {}".format(i))
        s_p = next(gen_p)

        # XXX: results are not not cached, but recomputed for each i
        gen_r = eliminate_band(a, b, c, d)
        # print("j: [{}, {}]".format(i+1, end))
        for k in range(i+1, end): # remaining rows for upwards elimination
            s_r = next(gen_r)

        mtx_check = np.array([[s_p[1], s_p[2]], 
                              [s_r[2], s_r[1]]])
        # mtx_check = np.array([[s_p[0], s_p[1]], [s_r[1], s_r[0]]])
        mtx_conds.append(np.linalg.cond(mtx_check))
        mtx_dets.append(np.abs(np.linalg.det(mtx_check)))

    return mtx_conds, mtx_dets

# %% check 2x2 systems (ignoring values of lower and upper spike) for blocks
remainder = 512
part_conds = []
part_begin = 0
part_min_size = 20
part_max_size = 40

while remainder >= 40:
    mtx_conds, mtx_dets = split_block_reduce_inner_vals(
        a_fine, b_fine, c_fine, d_fine, part_begin, part_begin+part_max_size)
    conds_asc = np.argsort(mtx_conds)
    # conds_asc = np.flip(np.argsort(mtx_dets))
    
    # first row[i] with partition size >= min
    i_opt = np.argwhere(conds_asc >= part_min_size)[0][0]
    # row[i+1] contains lower spike of compraed block
    offset = conds_asc[i_opt]+3

    # print([part_begin, part_begin+offset+1])
    part_conds.append([part_begin, part_begin+offset])
    part_begin += offset
    remainder -= offset

part_conds[-1] = [472, 512]

# %%
for p in part_conds:
    print(p[1] - p[0])
    
# %%
x_fine_conds, mtx_coarse_conds, mtx_cond_coarse_conds = rpta.reduce_and_solve(
    2*len(part_conds), a_fine, b_fine, c_fine, d_fine, part_conds)

fre_conds = np.linalg.norm(
    x_fine_conds - x_fine) / np.linalg.norm(x_fine)


# %% Normally vs. uniformly distributed partition boundaries
n_samples = 1000
n_part_min = 24
n_part_max = 48

fre_random_uniform = []
cond_random_uniform = []
for sample in main_random(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                          n_samples, 'uniform', n_part_min, n_part_max):
        _, _fre_u, _, _cond_u, _ = sample
        fre_random_uniform.append(_fre_u)
        cond_random_uniform.append(_cond_u)

# %%
n_part_mean = 32
n_part_sd = 1

fre_random_normal = []
cond_random_normal = []
for sample in main_random(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
                          n_samples, 'normal', n_part_min, n_part_max,
                          n_part_mean, n_part_sd):
    _, _fre_n, _, _cond_n, _ = sample
    fre_random_normal.append(_fre_n)
    cond_random_normal.append(_cond_n)