#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:18:23 2022

@author: ferdinand
"""
import numpy as np
import partition, matrix, rpta

from math import ceil
from scipy.io import mmread
from main_random import main_random


# %% Generate linear system
# TODO: take seed as argument for solution vector (fixed for generated matrix)
def main_setup(mtx_id, N_fine):
    np.random.seed(0)
    a_fine, b_fine, c_fine = matrix.scipy_matrix_to_bands(
        mmread("../mtx/{:02d}-{}".format(mtx_id, N_fine)))

    # Solution
    mtx = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    x_fine = np.random.normal(3, 1, N_fine)

    # Right-hand side
    d_fine = np.matmul(mtx, x_fine)
    
    return a_fine, b_fine, c_fine, d_fine, x_fine


# %% Input parameters
N_fine = 2048
# N_fine = 512
M = 32
N_tilde = (ceil(N_fine / M)) * 2 # one reduction step
seed = 0
part_min, part_max = 32, 100


# %% Randomly generated blocks
# Shows condition of coarse system and FRE may differ (N=2048, n_samples=5000)
# ID,lim_lo,lim_hi,min_fre,cond_coarse
# 14,4374,1.460142e-04,3.880050e+16
# 14,3075,1.787247e-02,6.600094e+12

print('ID,lim_lo,lim_hi,min_fre,cond_coarse')
mtx_id = 14
n_samples = 5000

# Set up linear system
a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(mtx_id, N_fine)

# Minimize over FRE
fre1, mtx_coarse1, mtx_cond_coarse1 = main_random(
    mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, n_samples, part_min, part_max, 'fre', seed)

# Minimize over condition of the coarse system
fre2, mtx_coarse2, mtx_cond_coarse2 = main_random(
    mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, n_samples, part_min, part_max, 'cond', seed)


# %% Check diagonal dominance
