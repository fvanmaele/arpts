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

from main_static import main_static
from main_cond_coarse import main_cond_coarse
from main_random import main_random
from main_rows import main_rows

def cond(A):
    print(np.format_float_scientific(np.linalg.cond(A)))

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
import json
import glob
import matplotlib.pyplot as plt

# %%
N_fine = 512
mtx_id = 14
n_holes = 12

# %%
decoupled = glob.glob("../decoupled/{:0>2}/rhs1/mtx-{}-{}-decoupled-*.json".format(
    n_holes, mtx_id, N_fine))
decoupled.sort()

mtx_data = []
for d in decoupled:
    with open(d, 'r') as d_json:
        mtx_data.append(json.load(d_json))

conds  = np.array([d['condition'] for d in mtx_data])
maxacc = np.array([d['max_accuracy'] for d in mtx_data])
relres = np.array([d['residual'] / np.ones(512) for d in mtx_data])
rhs = np.array(mtx_data[0]['rhs'])  # fixed rhs for all samples

# %% samples of highest condition, excluding outliers to keep single order of magnitude
# (stddev as measure of numerical stability of algorithm)
n_max_samples = 100
idx_decoupled = list(reversed(np.argsort(conds)))[10:n_max_samples+10]

# %% Histogram on linear scale
#hist, bins, _ = plt.hist(conds[idx_decoupled], bins=50)
hist, bins, _ = plt.hist(conds, bins=50)

# %% Histogram on log scale
logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))
plt.hist(conds[idx_decoupled], bins=logbins)
plt.xscale('log')
#plt.savefig("mtx-{}-{:0>2}-{}-cond".format(mtx_id, n_holes, len(decoupled)), dpi=108)

# %%
mtx_decoupled = glob.glob("../decoupled/{:0>2}/mtx-{}-{}-decoupled-*.mtx".format(
    n_holes, mtx_id, N_fine))
mtx_decoupled.sort()

# %%
fre_dec = []
fre_static = []
M_range = range(32, 65)

for i in idx_decoupled:
#for i, m in enumerate(mtx_decoupled):
    mtx = mmread(mtx_decoupled[i])
    print(mtx_decoupled[i], ", cond: ", mtx_data[i]['condition'])
    a_fine_m, b_fine_m, c_fine_m = matrix.scipy_matrix_to_bands(mtx)
    x_fine_m = np.array(mtx_data[i]['solution'])

    # Convert holes (1-indexed) into partition (0-indexed)
    holes_0idx = np.array(mtx_data[i]['sample_1idx']) - 1
    partition_decoupled = [[0, holes_0idx[0]]]
    for prev, curr in zip(holes_0idx, holes_0idx[1:]):
        partition_decoupled.append([prev, curr])

    partition_decoupled.append([holes_0idx[-1], N_fine])

    x_rpta_dec, mtx_coarse_dec, mtx_cond_coarse_dec = rpta.reduce_and_solve(
        a_fine_m, b_fine_m, c_fine_m, rhs, partition_decoupled, pivoting='scaled_partial')
    fre_dec.append(np.linalg.norm(x_rpta_dec - x_fine_m) / np.linalg.norm(x_fine_m))
    
    # Comparison with static partition  
    fre_static_i = []
    for M in M_range:
        partition_static = partition.generate_static_partition(N_fine, M)
    
        x_rpta_static, mtx_coarse_static, mtx_cond_coarse_static = rpta.reduce_and_solve(
            a_fine_m, b_fine_m, c_fine_m, rhs, partition_static, pivoting='scaled_partial')

        fre_static_i.append(np.linalg.norm(x_rpta_static - x_fine_m) / np.linalg.norm(x_fine_m))    
        print("fre (decoupled): {}, fre (M = {}): {}".format(fre_dec[-1], M, fre_static_i[-1]))
    
    fre_static.append(fre_static_i)

# %%
plt.yscale('log')
plt.plot(range(0, n_max_samples), fre_dec, 'o')
plt.plot(range(0, n_max_samples), [fre_static[id][0] for id in range(0, len(fre_static))], 'o')

# %%
fre_idx = [0]
fre_lab = ["D"]
fre_mean = [np.mean(fre_dec)]
fre_std = [np.std(fre_dec)]
fre_static_t = np.array(fre_static).T

for k, M in enumerate(M_range):
    fre_mean.append(np.mean(fre_static_t[k]))
    fre_std.append(np.std(fre_static_t[k]))
    fre_idx.append(k+1)
    fre_lab.append(str(M))

plt.figure(figsize=(10,4))
plt.title("mtx_id {} - n_holes {} - n_samples {} - rhs {}".format(mtx_id, n_holes, len(mtx_decoupled), 1))
plt.xticks(ticks=fre_idx, labels=fre_lab)
plt.yscale('log')
plt.errorbar(fre_idx, fre_mean, fre_std, linestyle='None', marker='o', capsize=3)
plt.tight_layout()
plt.savefig("mtx-{}-{:0>2}-{}-rhs1".format(mtx_id, n_holes, len(mtx_decoupled)), dpi=108)

# %%
plt.yscale('log')
plt.plot(fre_static_t[1])  # forward error for M = 33 on 100 samples

# %%
plt.yscale('log')
plt.plot(conds[idx_decoupled])  # condition number for 100 samples
