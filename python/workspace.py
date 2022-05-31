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

# %% Generate coarse system
#for sample in main_static(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, [32], min_part=6):
#    x_fine_rptapp, M, fre, mtx_coarse, mtx_cond_coarse, rpta_partition = sample

# %%
A33 = A[0:33, 0:33]
a33, b33, c33 = matrix.numpy_matrix_to_bands(A33)
d33 = d_fine[0:33]

a2_33, b2_33, c2_33, d2_33 = [np.zeros(2)]*4
rpta.rptapp_reduce(a33, b33, c33, d33, a2_33, b2_33, c2_33, d2_33, [[0, 33]])

# %%
#A32 = A[0:32, 0:32]
#a32, b32, c32 = matrix.numpy_matrix_to_bands(A32)
#d32 = d_fine[0:32]
#
#a2_32, b2_32, c2_32, d2_32 = [np.zeros(2)]*4
#rpta.rptapp_reduce(a32, b32, c32, d32, a2_32, b2_32, c2_32, d2_32, [[0, 32]])
#
