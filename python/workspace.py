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


# %% Linear system
mtx_id = 14
N_fine = 512

A_sp = mmread('{}/mtx/{:02d}-{}'.format(source_dir, mtx_id, N_fine))
try:
    A = np.asarray(A_sp.todense())
except AttributeError:
    A = np.asarray(A_sp)

# %%
a_fine, b_fine, c_fine = matrix.numpy_matrix_to_bands(A)
x_fine = np.random.normal(3, 1, N_fine)
d_fine = np.matmul(A, x_fine)

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


def eliminate_band_reversed(a, b, c, d, pivoting='scaled_partial'):
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    c_rev = list(reversed(c))
    d_rev = list(reversed(d))
    
    yield from eliminate_band(c_rev, b_rev, a_rev, d_rev, pivoting)


# %%
def print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    # M = len(range(begin, end))
    a = a_fine[begin:end]
    b = b_fine[begin:end]
    c = c_fine[begin:end]
    d = d_fine[begin:end]
    spikes = []

    for i, s_p in enumerate(eliminate_band(a, b, c, d, pivoting)):
        # a, b, c, d = s_p
        print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>5.1f} {:>20.6e}".format(
            begin+i+1, s_p[0], s_p[1], s_p[2], s_p[3], s_p[4]))
        spikes.append(s_p[0])
        
    return np.array(spikes)


def print_upwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    M = len(range(begin, end))
    a_rev = list(reversed(a_fine[begin:end]))
    b_rev = list(reversed(b_fine[begin:end]))
    c_rev = list(reversed(c_fine[begin:end]))
    d_rev = list(reversed(d_fine[begin:end]))
    spikes = []

    for i, s_r in enumerate(eliminate_band(c_rev, b_rev, a_rev, d_rev, pivoting)):
        # c, b, a, d = s_r
        print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>5.1f} {:>20.6e}".format(
            M-(i+1), s_r[2], s_r[1], s_r[0], s_r[3], s_r[4]))
        spikes.append(s_r[2])
        
    return np.array(list(reversed(spikes)))


# %% Approximate solutions
# For partial pivoting, the value of the spikes are monotically decreasing.
# Use this to decouple the system by setting the spikes (after a certain amount
# of steps) to zero.
def partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, begin, end, 
                                 plateau_min=10, plateau_max=100, min_change=1e-8):
    a = a_fine[begin:end]
    b = b_fine[begin:end]
    c = c_fine[begin:end]
    d = d_fine[begin:end]

    prev_spike = 0
    spike = 0
    spike_eq_counter = 0
    spike_eq_idx = [] # candidate row indices for next block
    spike_eq_idx_prev = []

    # partial pivoting: spike is montonically decreasing
    for i, s_p in enumerate(eliminate_band(a, b, c, d, pivoting='partial')):
        # print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>5.1f} {:>20.6e}".format(
        #     begin+i+1, s_p[0], s_p[1], s_p[2], s_p[3], s_p[4]))

        spike = s_p[0]
        if abs(spike - prev_spike) < min_change:
            spike_eq_counter += 1
            spike_eq_idx.append(begin+i)
        else:
            spike_eq_counter = 1
            spike_eq_idx_prev = np.copy(spike_eq_idx) # save previous plateau in case of early termination
            spike_eq_idx = [begin+i] # new plateau of nearly identical spikes

        prev_spike = spike
        if spike_eq_counter >= plateau_max:
            break # XXX: wrong break here?

    if len(spike_eq_idx) < plateau_min:
        #spike_eq_idx = spike_eq_idx_prev + spike_eq_idx
        spike_eq_idx = np.hstack((spike_eq_idx_prev, spike_eq_idx))

    return (spike_eq_idx, spike)

def partition_by_min_spike_value_loop(a_fine, b_fine, c_fine, d_fine, plateau_min=10, plateau_max=100, min_change=1e-8):
    n = len(a_fine)
    begin = 0

    while True:
        spidx, _ = partition_by_min_spike_value(
            a_fine, b_fine, c_fine, d_fine, begin, min(n, begin+plateau_max), plateau_min, plateau_max, min_change)
        
        spikes = []
        for idx in spidx:
            _, spike = partition_by_min_spike_value(
                a_fine, b_fine, c_fine, d_fine, idx, min(n, idx+plateau_max), plateau_min, plateau_max, min_change)
            spikes.append(spike)
        
        spidx_min = spidx[np.argmin(np.abs(spikes))]
        part = [begin, spidx_min]
        begin = spidx_min
        
        if part[1] >= n-1:
            break

        yield part

# %%
part = []
for p in partition_by_min_spike_value_loop(a_fine, b_fine, c_fine, d_fine):
    print(p)

# %%
spidx, _ = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, 0, 100)

for i_cand in spidx:
    _, spike = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, i_cand, 100+i_cand)
    print("i =", i_cand, ", last =", spike)

# i = 98 , last = -3.548013189672729e-08

# %%
spidx, _ = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, 98, 198)

for i_cand in spidx:
    _, spike = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, i_cand, 100+i_cand)
    print("i =", i_cand, ", last =", spike)

# i = 190 , last = 1.548863040005428e-06
# i = 196 , last = 1.5387333029915054e-07

# %%
spidx, _ = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, 190, 250)

for i_cand in spidx:
    _, spike = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, i_cand, 60+spidx[0])
    print("i =", i_cand, ", last =", spike)

# i = 226 , last = 2.7210944422397176e-06

# %%
spidx, _ = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, 226, 326)

for i_cand in spidx:
    _, spike = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, i_cand, 100+spidx[0])
    print("i =", i_cand, ", last =", spike)
    
# i = 313 , last = 2.0628660814666753e-05

# %%
spidx, _ = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, 313, 413)

for i_cand in spidx:
    _, spike = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, i_cand, 100+spidx[0])
    print("i =", i_cand, ", last =", spike)

# i = 377, last = 9.257998e-05 [manual verification]

# %%
spidx, _ = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, 377, 512)

for i_cand in spidx:
    _, spike = partition_by_min_spike_value(a_fine, b_fine, c_fine, d_fine, i_cand, 135+spidx[0])
    print("i =", i_cand, ", last =", spike)

# i = 377, last = 9.257998e-05 [manual verification]
# i = 450, last = 4.36160003e-06 [manual verification]


# %%
part = [[0,98], [98,190], [190,226], [226,313], [313,377], [377,450], [450, 512]]
for p in part:
    _ = print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, p[0], p[1], 'partial')
    print()

# %% Solve problem exactly
x_fine_approx, mtx_coarse_approx, mtx_cond_coarse_approx = rpta.reduce_and_solve(
    len(part)*2, a_fine, b_fine, c_fine, d_fine, part, pivoting='partial')

# %%
# TODO: verify this code (check that upwards and downwards elimination is correct)
# TODO: check this method for other matrices (and partial pivoting)
def split_block_reduce_inner_vals(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    mtx_conds, mtx_dets = [np.Inf], [0]
    a = a_fine[begin:end]
    b = b_fine[begin:end]
    c = c_fine[begin:end]
    d = d_fine[begin:end]
    
    gen_p = eliminate_band(a, b, c, d, pivoting)
    for i in range(begin+1, end-1):  # rows for downwards elimination, a[1] ... a[M-1]
        # print("i: {}".format(i))
        s_p = next(gen_p)
        gen_r = eliminate_band_reversed(a, b, c, d, pivoting)

        # print("k: [{}, {}]".format(i+1, end))
        # XXX: results are not not cached, but recomputed for each i
        for k in range(i+1, end): # remaining rows for upwards elimination
            s_r = next(gen_r)

        mtx_check = np.array([[s_p[1], s_p[2]], 
                              [s_r[2], s_r[1]]])
        # mtx_check = np.array([[s_p[0], s_p[1]], 
        #                       [s_r[1], s_r[0]]])

        mtx_conds.append(np.linalg.cond(mtx_check))
        mtx_dets.append(np.abs(np.linalg.det(mtx_check)))

    return mtx_conds, mtx_dets

# %% check 2x2 systems (ignoring values of lower and upper spike) for blocks
remainder = 512
part_conds = []
part_begin = 0
part_min_size = 20
part_max_size = 40
pivoting = 'partial'

while remainder >= part_max_size:
    mtx_conds, mtx_dets = split_block_reduce_inner_vals(
        a_fine, b_fine, c_fine, d_fine, 
        part_begin, part_begin+part_max_size, pivoting)
    conds_asc = np.argsort(mtx_conds)
    # conds_asc = np.flip(np.argsort(mtx_dets))
    
    # first row[i] with partition size >= min
    i_opt = np.argwhere(conds_asc >= part_min_size)[0][0]
    # row[i+1] contains lower spike of compared block
    offset = conds_asc[i_opt]+2

    # print([part_begin, part_begin+offset+1])
    part_conds.append([part_begin, part_begin+offset])
    part_begin += offset
    remainder -= offset

if 512 - part_conds[-1][1] < part_min_size:
    part_conds[-1] = [part_conds[-1][0], 512]
elif part_conds[-1][1] < 512:
    part_conds.append([part_conds[-1][1], 512])

#part_conds[-1] = [472, 512]
#part_conds.append([495, 512])

