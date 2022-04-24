#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:48:07 2022

@author: ferdinand
"""
import numpy as np
import sys
import matrix, partition, rpta


def main_setup(mtx_id, N_fine):
    np.random.seed(0)
    return matrix.generate_tridiag_system(
            mtx_id, N_fine, unif_low=-1, unif_high=1)


def main_rows(a_fine, b_fine, c_fine, d_fine, x_fine, 
              N_fine, lim_lo, lim_hi):
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
    mtx_id = int(sys.argv[1])
    N_fine = int(sys.argv[2])
    lim_lo = int(sys.argv[3]) # 10..36
    lim_hi = int(sys.argv[4]) # 20..72

    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(mtx_id, N_fine)
    main_rows(a_fine, b_fine, c_fine, d_fine, x_fine, N_fine, 
              lim_lo, lim_hi)