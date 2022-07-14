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
import matrix, partition, rpta, symmetric
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
try:
    A = np.asarray(A_sp.todense())
except AttributeError:
    A = np.asarray(A_sp)

# %%
a_fine, b_fine, c_fine = matrix.numpy_matrix_to_bands(A)
x_fine = np.random.normal(3, 1, N_fine)
d_fine = np.matmul(A, x_fine)


# %%
def print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    # M = len(range(begin, end))
    a = a_fine[begin:end]
    b = b_fine[begin:end]
    c = c_fine[begin:end]
    d = d_fine[begin:end]
    # spikes = []

    for i, s_p in enumerate(symmetric.eliminate_band_iter(a, b, c, d, pivoting), start=1):
        # a, b, c, d = s_p
        print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>5.1f} {:>20.6e}".format(
            begin+i, s_p[0], s_p[1], s_p[2], s_p[3], s_p[4]))
        # spikes.append(s_p[0])
        
    # return np.array(spikes)

def print_upwards_elimination(a_fine, b_fine, c_fine, d_fine, begin, end, pivoting):
    M = len(range(begin, end))
    a_rev = list(reversed(a_fine[begin:end]))
    b_rev = list(reversed(b_fine[begin:end]))
    c_rev = list(reversed(c_fine[begin:end]))
    d_rev = list(reversed(d_fine[begin:end]))
    # spikes = []

    for i, s_r in enumerate(symmetric.eliminate_band_iter(c_rev, b_rev, a_rev, d_rev, pivoting), start=2):
        # c, b, a, d = s_r
        print("{:2}: {:>20.6e} {:>20.6e} {:>20.6e} {:>5.1f} {:>20.6e}".format(
            M-i, s_r[2], s_r[1], s_r[0], s_r[3], s_r[4]))
        # spikes.append(s_r[2])
        
    # return np.array(list(reversed(spikes)))

# %%
print_downwards_elimination(a_fine, b_fine, c_fine, d_fine, 0, 32, 'partial')

# %%
print_upwards_elimination(a_fine, b_fine, c_fine, d_fine, 0, 32, 'partial')

# %%
a = a_fine[0:32]
b = b_fine[0:32]
c = c_fine[0:32]
d = d_fine[0:32]
b_down, c_down, d_down, s_down, a_up, b_up, d_up, s_up = symmetric.rpta_symmetric(a, b, c, d, pivoting='partial')

# %%