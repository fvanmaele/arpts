#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:52:41 2022

@author: ferdinand
"""
import numpy as np
import sys
import matrix, partition, rpta


def main_setup(mtx_id, N_fine):
    np.random.seed(0)
    return matrix.generate_tridiag_system(
            mtx_id, N_fine, unif_low=-1, unif_high=1)
    

def main_static(a_fine, b_fine, c_fine, d_fine, x_fine, N_fine, M):
    rpta_partition = partition.generate_static_partition(N_fine, M)
    N_coarse = len(rpta_partition)*2

#    print('ID,M,fre,cond_coarse')
    fre, cond_coarse = rpta.reduce_and_solve(
            N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
            rpta_partition, threshold=0)
    print("{},{},{:e},{:e}".format(mtx_id, M, fre, cond_coarse))

    return fre, cond_coarse


if __name__ == "__main__":
    mtx_id = int(sys.argv[1])
    N_fine = int(sys.argv[2])
    M = int(sys.argv[3])

    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(mtx_id, N_fine)
    main_static(a_fine, b_fine, c_fine, d_fine, x_fine, 
                N_fine, M)
