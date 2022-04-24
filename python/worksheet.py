#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:18:23 2022

@author: ferdinand
"""
from math import ceil
import numpy as np

import partition, matrix, rpta
import main_static, main_random, main_rows, main_cond_part, main_cond_coarse

# %% Input parameters
N_fine = 512
# M = 61
M = 32
N_tilde = (ceil(N_fine / M)) * 2 # one reduction step
#mtx_id = 9

# %% Sanity checks
for mtx_id in range(1, 21):
    print('Generating {} (N = {})'.format(mtx_id, N_fine))
    a, b, c = matrix.generate_tridiag(mtx_id, N_fine)
    assert(len(a) == N_fine)
    assert(len(b) == N_fine)
    assert(len(c) == N_fine)

# %% Partition with fixed-size blocks
for mtx_id in range(1, 21):
    main_static(mtx_id, N_fine, M)

# %%
static_partition = partition.generate_static_partition(N_fine, M)
N_coarse = len(static_partition)*2
print ('ID,M,fre,cond_coarse')

for mtx_id in [14]:
    np.random.seed(0)
    a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
        mtx_id, N_fine, unif_low=-1, unif_high=1)
    
    fre, cond_coarse = rpta.reduce_and_solve(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
        static_partition, threshold=0)
    print("{},{},{:e},{:e}".format(mtx_id, M, fre, cond_coarse))


# %% Test setting boundaries from original system
# det/argmax:   14,520,2.050581e-06,2.170328e+12
# cond/argmin:  14,520,2.050581e-06,2.170328e+12
print('ID,lim_lo,lim_hi,fre,cond_coarse')

for mtx_id in range(1, 21):
    # Generate fine system
    np.random.seed(0)
    a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
            mtx_id, N_fine, unif_low=-1, unif_high=1)
    errs, conds = [], []

    for lim_lo in range(10, 36):
        for lim_hi in range(20, 72):
            if lim_lo >= lim_hi: 
                continue

            rpta_partition = partition.generate_partition_func(
                a_fine, b_fine, c_fine, lim_lo, lim_hi, func=np.linalg.cond, argopt=np.argmin)
            N_coarse = len(rpta_partition)*2

            fre, cond_coarse = rpta.reduce_and_solve(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
                         rpta_partition, threshold=0)
            print("{},{},{},{:e},{:e}".format(
                    mtx_id, lim_lo, lim_hi, fre, cond_coarse))
            errs.append(fre)
            conds.append(cond_coarse)

    min_idx = np.argmin(errs)
    print('{},{},{:e},{:e}'.format(mtx_id, min_idx, errs[min_idx], conds[min_idx]))


# %% Test minimal condition of reduced system (condition for partition boundaries)
# Empirically, a difference of 5-10 between lim_lo and lim_hi leads to a good
# forward relative error. It also limits the performance impact, especially
# when redundant computations are done.
print('ID,lim_lo,lim_hi,fre,cond_coarse')

for mtx_id in range(1, 21):
    # Generate fine system\
    np.random.seed(0)
    a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
            mtx_id, N_fine, unif_low=-1, unif_high=1)

    for lim_lo in range(10, 36):
        for lim_hi in range(20, 72):
            if lim_lo >= lim_hi: 
                continue

            rpta_partition = rpta.rptapp_reduce_dynamic(
                    a_fine, b_fine, c_fine, d_fine, lim_lo, lim_hi, threshold=0)
            N_coarse = len(rpta_partition)*2

            fre, cond_coarse = rpta.reduce_and_solve(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
                         rpta_partition, threshold=0)
            print("{},{},{},{:e},{:e}".format(
                    mtx_id, lim_lo, lim_hi, fre, cond_coarse))


# %% Test random sampling
print('ID,n_sample,min_fre,cond_coarse')

for mtx_id in range(1,21):
    # Generate fine system
    np.random.seed(0)
    a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
            mtx_id, N_fine, unif_low=-1, unif_high=1)
    
#    n_samples = 50
    n_samples = 100
    errs, conds = [], []
    part_min, part_max = 16, 64


    for n in range(0, n_samples):
        rpta_partition = partition.generate_random_partition(N_fine, part_min, part_max)
#        print(rpta_partition)
        N_coarse = len(rpta_partition)*2
        
        fre, cond_coarse = rpta.reduce_and_solve(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
                                            rpta_partition, threshold=0)
        errs.append(fre)
        conds.append(cond_coarse)
        
#        print("{},{},{:e},{:e}".format(mtx_id, n, fre, cond_coarse), file=stderr)
    
    min_idx = np.argmin(errs)
    print('{},{},{:e},{:e}'.format(mtx_id, min_idx, errs[min_idx], conds[min_idx]))