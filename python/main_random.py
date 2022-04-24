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

# TODO: accept seed as parameter?
def main_random(a_fine, b_fine, c_fine, d_fine, x_fine, 
                N_fine, n_samples, part_min, part_max):
    np.random.seed(0)
    errs, conds = [], []

    for n in range(0, n_samples):
        rpta_partition = partition.generate_random_partition(
                N_fine, part_min, part_max)
        N_coarse = len(rpta_partition)*2
        
        fre, cond_coarse = rpta.reduce_and_solve(
                N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
                rpta_partition, threshold=0)
        errs.append(fre)
        conds.append(cond_coarse)
#        print("{},{},{:e},{:e}".format(mtx_id, n, fre, cond_coarse), file=sys.stderr)
    
    min_idx = np.argmin(errs)
    print('{},{},{:e},{:e}'.format(mtx_id, min_idx, errs[min_idx], conds[min_idx]))
    
    return errs[min_idx], conds[min_idx]


if __name__ == "__main__":
    mtx_id = int(sys.argv[1])
    N_fine = int(sys.argv[2])
    n_samples = int(sys.argv[3])
    part_min = int(sys.argv[4])
    part_max = int(sys.argv[5])
    
    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(mtx_id, N_fine)
    main_random(a_fine, b_fine, c_fine, d_fine, x_fine, N_fine, 
                n_samples, part_min, part_max)