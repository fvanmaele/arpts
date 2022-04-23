#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:18:23 2022

@author: ferdinand
"""
from math import ceil
import numpy as np
from sys import stderr

import partition
import matrix
import rpta


# %%
def reduce_run(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, partition, threshold=0):
    # Reduce to coarse system
    a_coarse = np.zeros(N_coarse)
    b_coarse = np.zeros(N_coarse)
    c_coarse = np.zeros(N_coarse)
    d_coarse = np.zeros(N_coarse)
    
    rpta.rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse, d_coarse,
                       partition, threshold=0)
    mtx_coarse = matrix.bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)
    
    # Plot coarse system
    # partition.plot_coarse_system(mtx_coarse, "Condition: {:e}".format(mtx_cond_coarse))
    
    try:
        x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
        x_fine_rptapp = rpta.rptapp_substitute(
                a_fine, b_fine, c_fine, d_fine, x_coarse, rpta_partition, threshold=0)

        mtx_cond_coarse = np.linalg.cond(mtx_coarse)
        fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)

    except np.linalg.LinAlgError:
        print("warning: Singular matrix detected", file=stderr)
        fre, mtx_cond_coarse = np.Inf, np.Inf
        
    return fre, mtx_cond_coarse


# %% Input parameters
N_fine = 512
# M = 61
M = 32
N_tilde = (ceil(N_fine / M)) * 2 # one reduction step
#mtx_id = 9

# %% TODO: Partition with fixed-size blocks
# dyn_partition, mtx_cond, mtx_cond_partmax, mtx_cond_partmax_dyn = rptapp_generate_partition(
#     a_fine, b_fine, c_fine, M, n_halo, k_max_up, k_max_down)
static_partition = partition.generate_static_partition(N_fine, M)


# %% TODO: Test setting boundaries from original system
#    rpta_partition = partition.generate_partition_func(
#        a_fine, b_fine, c_fine, 16, 64, func=np.linalg.det, argopt=np.argmax)


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

            fre, cond_coarse = reduce_run(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
                         rpta_partition, threshold=0)
            print("{},{},{},{:e},{:e}".format(
                    mtx_id, lim_lo, lim_hi, fre, cond_coarse))


# %% Test random sampling
print('ID,n_sample,min_fre,cond_coarse')

for mtx_id in range(1,21): # XXX: catch singular matrix for mtx_id == 15
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
        
        fre, cond_coarse = reduce_run(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
                                      rpta_partition, threshold=0)
        errs.append(fre)
        conds.append(cond_coarse)
        
#        print("{},{},{:e},{:e}".format(mtx_id, n, fre, cond_coarse), file=stderr)
    
    min_idx = np.argmin(errs)
    print('{},{},{:e},{:e}'.format(mtx_id, min_idx, errs[min_idx], conds[min_idx]))