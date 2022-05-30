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
for sample in main_static(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, [33], min_part=6):
    x_fine_rptapp, M, fre, mtx_coarse, mtx_cond_coarse, rpta_partition = sample