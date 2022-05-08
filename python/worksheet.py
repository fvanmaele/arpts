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
mtx_id = 14
N_fine = 512
M = 32
N_tilde = (ceil(N_fine / M)) * 2 # one reduction step
part_min, part_max = 32, 64  # TODO: use numbers from parameter study
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
# Shows condition of coarse system and FRE may differ (N=2048, n_samples=5000)
# ID,lim_lo,lim_hi,min_fre,cond_coarse
# 14,4374,1.460142e-04,3.880050e+16
# 14,3075,1.787247e-02,6.600094e+12
print('ID,N,lim_lo,lim_hi,min_fre,cond_coarse')
min_fre = np.Inf
min_fre_part = []

for n in range(0, n_samples):
    _sol, _fre, _coarse, _cond_coarse, _part = main_random(
        mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, part_min, part_max)
    if _fre < min_fre:
        min_fre = _fre
        min_fre_part = _part

part1 = min_fre_part

# %%
print('ID,N,lim_lo,lim_hi,min_fre,cond_coarse')
min_cond = np.Inf
min_cond_part = []

for n in range(0, n_samples):
    _sol, _fre, _coarse, _cond_coarse, _part = main_random(
        mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, part_min, part_max)
    if _cond_coarse < min_cond:
        min_cond = _cond_coarse
        min_cond_part = _part

part2 = min_cond_part

# %% Blocks optimized on minimal condition of the coarse system
print('ID,N,lim_lo,lim_hi,fre,cond_coarse')
sol3, fre3, mtx_coarse3, mtx_cond_coarse3, part3 = main_cond_coarse(
    mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 32, 64)

# %% Blocks of fixed size
print('ID,N,M,fre,cond_coarse')
sol4, fre4, mtx_coarse4, mtx_cond_coarse4, part4 = main_static(
    mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 48)

# %% Check behavior with other right-hand sides
def check_partition_over_samples(target_partition, mean=0, stddev=1):
    N_fine = target_partition[-1][1]
    N_coarse = len(target_partition)*2
    x_fine_new = np.random.normal(mean, stddev, N_fine)
    d_fine_new = np.matmul(mtx, x_fine_new)

    x_fine_rptapp_new, mtx_coarse_new, mtx_cond_coarse_new = rpta.reduce_and_solve(
        N_coarse, a_fine, b_fine, c_fine, d_fine_new, target_partition, threshold=0)
    fre_new = np.linalg.norm(x_fine_rptapp_new - x_fine_new) / np.linalg.norm(x_fine_new)

    return fre_new

# %% Random, minimized over FRE for 1 sample
trials_part1 = [None]*1000
for k in range(0, 1000):
    if k % 20 == 0:
        print("trial #{}, part1".format(k))
    trials_part1[k] = check_partition_over_samples(part1, 3, 1)
plt.plot(trials_part1)

# %% Random, minimized over condition of coarse system
trials_part2 = [None]*1000
for k in range(0, 1000):
    if k % 20 == 0:
        print("trial #{}, part2".format(k))
    trials_part2[k] = check_partition_over_samples(part2, 3, 1)
plt.plot(trials_part2)

# %% Boundaries computed during reduce step, minimized over condition of coarse system
trials_part3 = [None]*1000
for k in range(0, 1000):
    if k % 20 == 0:
        print("trial #{}, part3".format(k))
    trials_part3[k] = check_partition_over_samples(part3, 3, 1)
plt.plot(trials_part3)

# %% Blocks of fixed size
trials_part4 = [None]*1000
for k in range(0, 1000):
    if k % 20 == 0:
        print("trial #{}, part4".format(k))
    trials_part4[k] = check_partition_over_samples(part4, 3, 1)
plt.plot(trials_part4)

# %%
print(describe(trials_part1))
print()
print(describe(trials_part2))
print()
print(describe(trials_part3))
print()
print(describe(trials_part4))

