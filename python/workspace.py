#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 30 13:06:27 2022

@author: fvanmaele
"""

# %%
import os

home_dir = os.environ['HOME']
source_dir = '{}/source/repos/arpts'.format(home_dir)
os.chdir('{}/python'.format(source_dir))

# %%
import matrix, partition, rpta, symmetric
import numpy as np

from scipy.io import mmread

# %%
from main_static import main_static
from main_cond_coarse import main_cond_coarse
from main_random import main_random
from main_rows import main_rows


# %% Linear system
mtx_id = 26
N_fine = 512

A_sp = mmread('{}/mtx/{:02d}-{}'.format(source_dir, mtx_id, N_fine))
try:
    A = np.asarray(A_sp.todense())
except AttributeError:
    A = np.asarray(A_sp)

a_fine, b_fine, c_fine = matrix.numpy_matrix_to_bands(A)
x_fine = np.random.normal(3, 1, N_fine)
d_fine = np.matmul(A, x_fine)

# %%
def cond(A):
    print(np.format_float_scientific(np.linalg.cond(A)))

# %%
def print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    a = a_fine[begin:end]
    b = b_fine[begin:end]
    c = c_fine[begin:end]
    d = d_fine[begin:end]

    s_lower, b_lower, c_lower, d_lower = symmetric.eliminate_band_expand(a, b, c, d, pivoting)

    for i in range(0, len(s_lower)):
        print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>20.6e}".format(
            begin+i, s_lower[i], b_lower[i], c_lower[i], d_lower[i]))
    
    # for i, s_p in enumerate(symmetric.eliminate_band_iter(a, b, c, d, pivoting)):
    #     print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>5.1f} {:>20.6e}".format(
    #         begin+i, s_p[0], s_p[1], s_p[2], s_p[3], s_p[4]))
        

def print_upwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    a_rev = list(reversed(a_fine[begin:end]))
    b_rev = list(reversed(b_fine[begin:end]))
    c_rev = list(reversed(c_fine[begin:end]))
    d_rev = list(reversed(d_fine[begin:end]))

    s_upper, b_upper, a_upper, d_upper = symmetric.eliminate_band_expand(c_rev, b_rev, a_rev, d_rev, pivoting)
    a_upper_rev = list(reversed(a_upper))
    b_upper_rev = list(reversed(b_upper))
    s_upper_rev = list(reversed(s_upper))
    d_upper_rev = list(reversed(d_upper))
    
    for i in range(0, len(a_upper)):
        print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>20.6e}".format(
            begin+i, a_upper_rev[i], b_upper_rev[i], s_upper_rev[i], d_upper_rev[i]))
    
    # for i, s_r in enumerate(symmetric.eliminate_band_iter(c_rev, b_rev, a_rev, d_rev, pivoting), start=1):
    #     print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>5.1f} {:>20.6e}".format(
    #         M-i, s_r[2], s_r[1], s_r[0], s_r[3], s_r[4]))


# %%
# print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, 0, 32, 'partial')

# %%
# print_upwards_elimination(a_fine, b_fine, c_fine, d_fine, 0, 32, 'partial')

# %%
static_partition = partition.generate_static_partition(512, 33)
x_fine_rpta, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
    a_fine, b_fine, c_fine, d_fine, static_partition, pivoting='scaled_partial')
fre_rpta = np.linalg.norm(x_fine_rpta - x_fine) / np.linalg.norm(x_fine)
fre_rpta

# %%
# symmetric.rpta_symmetric(a_fine, b_fine, c_fine, d_fine, static_partition, 'partial')
x_fine_symm, _, _ = symmetric.rpta_symmetric(a_fine, b_fine, c_fine, d_fine, static_partition, 'scaled_partial')
fre_symm = np.linalg.norm(x_fine_symm - x_fine) / np.linalg.norm(x_fine)
fre_symm

# %%
max_part = 40
min_part = 20
n_sliding_window = 4
dynpart_interface = []

def part_sliding_window(a_fine, b_fine, c_fine, part_begin, min_part, max_part, n_sliding_window):
    argmin_spike_rel_sum = [None, None]
    begin_offset = min_part-1
    min_spike_rel_sum = np.Inf
    
    for k in range(part_begin+begin_offset, part_begin+max_part+n_sliding_window):
        for ks in range(k+1, k+1+n_sliding_window):  # k+1: upper spike for part P, lower spike for part P+1
            part_upper_spike_rel = abs(c_fine[k]) / np.max([abs(a_fine[k]), abs(b_fine[k]), abs(c_fine[k])])
            next_part_lower_spike_rel = abs(a_fine[ks]) / np.max([abs(a_fine[ks]), abs(b_fine[ks]), abs(c_fine[ks])])
            spike_rel_sum = part_upper_spike_rel + next_part_lower_spike_rel
            
            if spike_rel_sum < min_spike_rel_sum:
                min_spike_rel_sum = spike_rel_sum
                argmin_spike_rel_sum = [k, ks]
                print(k, ks, min_spike_rel_sum)
    
    return argmin_spike_rel_sum

# %%
P = part_sliding_window(a_fine, b_fine, c_fine, 0, min_part, max_part, n_sliding_window)
dynpart_interface.append(P)

# %%
# TODO: iterate until partition M-1, then append
part_begin = dynpart_interface[-1][1]
P = part_sliding_window(a_fine, b_fine, c_fine, part_begin, min_part, max_part, n_sliding_window)
dynpart_interface.append(P)
dynpart_interface

# %%
dynpart_interface = [[35, 39],
 [80, 84],
 [115, 118],
 [144, 148],
 [189, 193],
 [228, 232],
 [263, 267],
 [286, 289],
 [332, 334],
 [373, 374],
 [393, 395],
 [437, 441],
 [463, 465],
 [499, 502]]

# TODO: Convert interfaces to coarse system
# dynpart_sparse = []
# part_begin = 0
# for P in dynpart_sparse:
#     dynpart_sparse.append([part_begin, P[0]])
#     dynpart_sparse.append([])
