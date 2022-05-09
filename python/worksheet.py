#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:18:23 2022

@author: ferdinand
"""
import numpy as np
import partition, matrix, rpta
import matplotlib.pyplot as plt

from math import ceil
from scipy.io import mmread
from scipy.stats import describe

from main_random import main_random
from main_cond_coarse import main_cond_coarse
from main_static import main_static
from main_rows import main_rows

# %% Input parameters
mtx_id = 13
N_fine = 512
M = 32
N_tilde = (ceil(N_fine / M)) * 2 # one reduction step
n_samples = 1000

# %% Number generator
seed = 0
np.random.seed(seed)

# %% Set up linear system
a_fine, b_fine, c_fine = matrix.scipy_matrix_to_bands(
    mmread("../mtx/{:02d}-{}".format(mtx_id, N_fine)))
mtx = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)

x_fine = np.random.normal(3, 1, N_fine)
#x_fine = np.random.uniform(-1, 1, N_fine)
d_fine = np.matmul(mtx, x_fine)  # Right-hand side


# %% Randomly generated blocks
# TODO: use more descriptive variable names, include coarse matrix
# Shows condition of coarse system and FRE may differ (N=2048, n_samples=5000)
# ID,lim_lo,lim_hi,min_fre,cond_coarse
# 14,4374,1.460142e-04,3.880050e+16
# 14,3075,1.787247e-02,6.600094e+12
min_fre1 = np.Inf
min_fre_part1 = []
min_cond1 = np.Inf
min_cond_part1 = []

for sample in main_random(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                          1000, 32, 100):
    _sol, _fre, _coarse, _cond_coarse, _part = sample

    if _fre < min_fre1:
        min_fre1 = _fre
        min_fre_part1 = _part
        
    if _cond_coarse < min_cond1:
        min_cond1 = _cond_coarse
        min_cond_part1 = _part


# %% Partition with minimal condition of the coarse system
# TODO: use more descriptive variable names, include coarse matrix
min_fre2 = np.Inf
min_cond2 = np.Inf
min_fre_part2 = []
min_cond_part2 = []

for sample in main_cond_coarse(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                               list(range(16,41)), list(range(22,73)), 6):
    _sol, _, _, _fre, _coarse, _cond_coarse, _part = sample
    
    if _fre < min_fre2:
        min_fre2 = _fre
        min_fre_part2 = _part
        
    if _cond_coarse < min_cond2:
        min_cond2 = _cond_coarse
        min_cond_part2 = _part

        
# %% Blocks of fixed size
# TODO: use more descriptive variable names, include coarse matrix
min_fre3 = np.Inf
min_fre_part3 = []
min_cond3 = np.Inf
min_cond_part3 = []

# x_fine_rptapp, M, fre, mtx_coarse, mtx_cond_coarse, rpta_partition
# main_static(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, M_range, min_part):
for sample in main_static(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
                          list(range(16,65)), 6):
    _sol, _, _fre, _coarse, _cond_coarse, _part = sample
    
    if _fre < min_fre3:
        min_fre3 = _fre
        min_fre_part3 = _part
        
    if _cond_coarse < min_cond3:
        min_cond3 = _cond_coarse
        min_cond_part3 = _part


# %% Check behavior with other right-hand sides
# TODO: use more descriptive variable names, include coarse matrix
def check_partition_over_samples(mtx, target_partition, mean=0, stddev=1):
    N_fine = target_partition[-1][1]
    N_coarse = len(target_partition)*2
    x_fine_new = np.random.normal(mean, stddev, N_fine)
    d_fine_new = np.matmul(mtx, x_fine_new)

    x_fine_rptapp_new, mtx_coarse_new, mtx_cond_coarse_new = rpta.reduce_and_solve(
        N_coarse, a_fine, b_fine, c_fine, d_fine_new, target_partition, threshold=0)
    fre_new = np.linalg.norm(x_fine_rptapp_new - x_fine_new) / np.linalg.norm(x_fine_new)

    return fre_new


# %%
def run_trials(mtx, part, label, mean=0, stddev=1, n_trials=1000):
    trials = [None]*n_trials
    for k in range(0, n_trials):
        if k % 20 == 0:
            print("trial #{}, {}".format(k, label))
        trials[k] = check_partition_over_samples(mtx, part, mean, stddev)

    return trials

# %% Random, minimized over FRE for 1 sample
trials_1 = run_trials(mtx, min_fre_part1, "random + fre", 3, 1)

# %% Random, minimized over condition of coarse system
trials_2 = run_trials(mtx, min_cond_part1, "random + cond", 3, 1)

# %% Boundaries computed during reduce step, minimized over FRE for 1 sample
trials_3 = run_trials(mtx, min_fre_part2, "coarse + fre", 3, 1)

# %% As above, but minimized over condition of the coarse system
trials_4 = run_trials(mtx, min_cond_part2, "coarse + cond", 3, 1)

# %% Blocks of fixed size
trials_5 = run_trials(mtx, min_fre_part3, "static + fre", 3, 1)

# %%
trials_6 = run_trials(mtx, min_cond_part3, "static + cond", 3, 1)

# %%
fig, axs = plt.subplots(3, 2)
axs[0, 0].bar(list(range(1,1001)), trials_1)
axs[0, 0].set_title('Random, min_fre')
axs[0, 1].bar(list(range(1,1001)), trials_2)
axs[0, 1].set_title('Random, min_cond')
axs[1, 0].bar(list(range(1,1001)), trials_3)
axs[1, 0].set_title('Reduce, min_fre')
axs[1, 1].bar(list(range(1,1001)), trials_4)
axs[1, 1].set_title('Reduce, min_cond')
axs[2, 0].bar(list(range(1,1001)), trials_5)
axs[2, 0].set_title('Static, min_fre')
axs[2, 1].bar(list(range(1,1001)), trials_6)
axs[2, 1].set_title('Static, min_cond')

for ax in axs.flat:
    ax.set(xlabel='n_sample', ylabel='FRE')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()

# %%
print(describe(trials_1))
print()
print(describe(trials_2))
print()
print(describe(trials_3))
print()
print(describe(trials_4))
print()
print(describe(trials_5))
print()
print(describe(trials_6))

