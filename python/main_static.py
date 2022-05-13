#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:52:41 2022

@author: ferdinand
"""
import argparse
import numpy as np
import matrix, partition, rpta

from scipy.io import mmread


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


def main_setup(mtx_id, N_fine):
    a_fine, b_fine, c_fine = matrix.scipy_matrix_to_bands(
        mmread("../mtx/{:02d}-{}".format(mtx_id, N_fine)))

    # Solution
    mtx = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    x_fine = np.random.normal(3, 1, N_fine)

    # Right-hand side
    d_fine = np.matmul(mtx, x_fine)
    
    return a_fine, b_fine, c_fine, d_fine, x_fine


def main_static(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, M_range, min_part):
    for M in M_range:
        rpta_partition = partition.generate_static_partition(N_fine, M, min_part)
        N_coarse = len(rpta_partition)*2
    
        x_fine_rptapp, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
            N_coarse, a_fine, b_fine, c_fine, d_fine, rpta_partition, threshold=0)
    
        if x_fine_rptapp is not None:
            fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
        else:
            fre = np.Inf

        yield x_fine_rptapp, M, fre, mtx_coarse, mtx_cond_coarse, rpta_partition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("M")
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    parser.add_argument("--min-size", type=int, default=5, help="minimum partition size")
    args = parser.parse_args()
    np.random.seed(args.seed)        

    # Range over M
    M_range = str_to_range(args.M, '-')
    # Generate linear system
    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(args.mtx_id, args.N_fine)
    
    for sample in main_static(args.N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                              M_range, args.min_size):
        _, M, fre, _, mtx_cond_coarse, _ = sample
        print("{},{},{},{:e},{:e}".format(args.mtx_id, args.N_fine, M, fre, mtx_cond_coarse))
