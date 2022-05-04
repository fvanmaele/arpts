#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:47:59 2022

@author: ferdinand
"""
import argparse
import numpy as np
import matrix, partition, rpta

from scipy.io import mmread


def main_setup(mtx_id, N_fine):
    a_fine, b_fine, c_fine = matrix.scipy_matrix_to_bands(
        mmread("../mtx/{:02d}-{}".format(mtx_id, N_fine)))

    # Solution
    mtx = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    x_fine = np.random.normal(3, 1, N_fine)

    # Right-hand side
    d_fine = np.matmul(mtx, x_fine)
    
    return a_fine, b_fine, c_fine, d_fine, x_fine


def main_cond_coarse(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                     lim_lo, lim_hi):
    rpta_partition = rpta.rptapp_reduce_dynamic(a_fine, b_fine, c_fine, d_fine, 
                                                lim_lo, lim_hi, threshold=0)
    N_coarse = len(rpta_partition)*2

    x_fine_rptapp, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
        N_coarse, a_fine, b_fine, c_fine, d_fine, rpta_partition, threshold=0)

    if x_fine_rptapp is not None:
        fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
    else:
        fre = np.Inf

    print("{},{},{},{},{:e},{:e}".format(mtx_id, N_fine, lim_lo, lim_hi, fre, mtx_cond_coarse))
    # Return solution and coarse system for further inspection
    return x_fine_rptapp, fre, mtx_coarse, mtx_cond_coarse, rpta_partition


def str_to_range(str, delim='-'):
    assert(len(str) >= 1)
    s_range = str.split(delim)
    s_range = list(map(int, s_range))
    
    if len(s_range) == 2:
        assert(s_range[1] > s_range[0])
        s_range = range(s_range[0], s_range[1]+1)
    elif len(s_range) != 1:
        raise ValueError

    return s_range


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("lim_lo")
    parser.add_argument("lim_hi")
    parser.add_argument("--min-size", type=int, default=8, help="minimal block size")
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Range over lim_lo, lim_hi
    lim_lo_range = str_to_range(args.lim_lo, '-')
    lim_hi_range = str_to_range(args.lim_hi, '-')
    
    # Generate linear system
    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(args.mtx_id, args.N_fine)

    for lim_lo in lim_lo_range:
        for lim_hi in lim_hi_range:
            if lim_hi - lim_lo < args.min_size:
                continue
            main_cond_coarse(args.mtx_id, args.N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                             lim_lo, lim_hi)