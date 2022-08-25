#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:48:07 2022

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


def main_rows(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
              lim_lo_range, lim_hi_range, func, argopt, min_size, pivoting):
    for lim_lo in lim_lo_range:
        for lim_hi in lim_hi_range:
            if lim_hi - lim_lo < args.min_size:
                continue
            
            rpta_partition = partition.generate_partition_func(
                a_fine, b_fine, c_fine, lim_lo, lim_hi, func=func, argopt=argopt)

            x_fine_rptapp, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
                a_fine, b_fine, c_fine, d_fine, rpta_partition, pivoting=pivoting)
            
            if x_fine_rptapp is not None:
                fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
                Ax = matrix.bands_mv(a_fine, b_fine, c_fine, x_fine_rptapp)
                res = np.linalg.norm(Ax - d_fine) / np.linalg.norm(d_fine)
            else:
                fre = np.Inf
                res = np.Inf

            yield x_fine_rptapp, fre, res, mtx_coarse, mtx_cond_coarse, rpta_partition, lim_lo, lim_hi


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("lim_lo")
    parser.add_argument("lim_hi")
    parser.add_argument("func")
    parser.add_argument("--min-size", type=int, default=8, help="minimal block size")
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    parser.add_argument("--pivoting", type=str, default='scaled_partial', help="type of pivoting used")
    args = parser.parse_args()
    np.random.seed(args.seed)
    
    # Range over lim_lo, lim_hi
    lim_lo_range = str_to_range(args.lim_lo, '-')
    lim_hi_range = str_to_range(args.lim_hi, '-')

    # Function for blocks of boundaries
    if args.func == "det":
        func = np.linalg.det
        argopt = np.argmax        
    elif args.func == "cond":
        func = np.linalg.cond
        argopt = np.argmin
    else:
        raise ValueError("func must be 'det' or 'cond'")

    # Generate linear system    
    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(args.mtx_id, args.N_fine)
    
    for sample in main_rows(args.N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
                            lim_lo_range, lim_hi_range, func, argopt, args.min_size, args.pivoting):
        _, fre, res, _, mtx_cond_coarse, _, lim_lo, lim_hi = sample
        
        print("{},{},{},{},{:e},{:e},{:e}".format(
            args.mtx_id, args.N_fine, lim_lo, lim_hi, fre, res, mtx_cond_coarse))