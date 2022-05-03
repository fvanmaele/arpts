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


def main_rows(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
              lim_lo, lim_hi):
    rpta_partition = partition.generate_partition_func(
        a_fine, b_fine, c_fine, lim_lo, lim_hi, 
        func=np.linalg.cond, argopt=np.argmin)
    N_coarse = len(rpta_partition)*2
    
    fre, cond_coarse = rpta.reduce_and_solve(
            N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
            rpta_partition, threshold=0)
    print("{},{},{},{:e},{:e}".format(
            mtx_id, lim_lo, lim_hi, fre, cond_coarse))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("lim_lo", type=int)
    parser.add_argument("lim_hi", type=int)
    args = parser.parse_args()

    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(args.mtx_id, args.N_fine)
    main_rows(args.mtx_id, args.N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
              args.lim_lo, args.lim_hi)