#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:47:59 2022

@author: ferdinand
"""
import numpy as np
import sys
import matrix, rpta


# TODO: keep partial results (in eleminate_band) of coarse system and compute determinant 
# The observation here is that consecutive rows in the coarse system are, for a bad partitioning,
# nearly linearly independent. As such, we can optimize for this when constructing (rows of)
# the coarse system.
def rptapp_reduce_dynamic(a_fine, b_fine, c_fine, d_fine, part_min, part_max, threshold=0):
    N = len(a_fine)
    assert(part_min < part_max)
    assert(part_min > 0)
    assert(part_max < N-1)
    
    partition = []
    partition_begin = 0 # Index of first row (upper boundary) 
    
    while partition_begin + part_max < N:
        conds = []
        for offset in range(part_min, part_max):
            partition_end = min(partition_begin + offset, N-1)
            
            a_coarse_lower, b_coarse_lower, c_coarse_lower, d_coarse_lower = rpta.eliminate_band(
                a_fine[partition_begin:partition_end],
                b_fine[partition_begin:partition_end],
                c_fine[partition_begin:partition_end],
                d_fine[partition_begin:partition_end],
                threshold)
            c_coarse_upper, b_coarse_upper, a_coarse_upper, d_coarse_upper = rpta.eliminate_band(
                list(reversed(c_fine[partition_begin:partition_end])),
                list(reversed(b_fine[partition_begin:partition_end])),
                list(reversed(a_fine[partition_begin:partition_end])),
                list(reversed(d_fine[partition_begin:partition_end])),
                threshold)
            
            # Criterion: maximum determinant
            mtx = np.matrix([[b_coarse_upper, c_coarse_upper],
                             [a_coarse_lower, b_coarse_lower]])
            mtx_cond = np.linalg.cond(mtx)
            conds.append(mtx_cond)
            # print("{}, {}: |det| = {}".format(partition_begin, partition_end, mtx_det))

        conds_argmin = np.argmin(conds)
        # print("{}, {}: |det| (max) = {}".format(
        #     partition_begin, partition_begin + part_min + dets_argmax, dets[dets_argmax]))
        partition.append([partition_begin, partition_begin + part_min + conds_argmin])

        # Go to next partition
        partition_begin = min(partition_begin + part_min + conds_argmin, N)
    
    # Append last partition
    if partition_begin < N:
        partition.append([partition_begin, N])

    return partition


def main_cond_coarse(mtx_id, N_fine, lim_lo, lim_hi):
    # Generate fine system\
    np.random.seed(0)
    a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
            mtx_id, N_fine, unif_low=-1, unif_high=1)

    rpta_partition = rpta.rptapp_reduce_dynamic(
        a_fine, b_fine, c_fine, d_fine, lim_lo, lim_hi, threshold=0)
    N_coarse = len(rpta_partition)*2

    fre, cond_coarse = rpta.reduce_and_solve(N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, 
            rpta_partition, threshold=0)
    print("{},{},{},{:e},{:e}".format(
            mtx_id, lim_lo, lim_hi, fre, cond_coarse))
    

if __name__ == "__main__":
    mtx_id = int(sys.argv[1])
    N_fine = int(sys.argv[2])
    lim_lo = int(sys.argv[3])
    lim_hi = int(sys.argv[4])

    main_cond_coarse(mtx_id, N_fine, lim_lo, lim_hi)