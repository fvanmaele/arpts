#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 11:47:47 2022

@author: ferdinand
"""
from math import ceil
import numpy as np
import sys

import partition, matrix, rpta, tridiag


# TODO: sufficient to compare condition locally when merge candidate is
# ambiguous? (instead of for the full blocks)
# XXX: rename to tridiag_merge_partition or generalize
def merge_partition(mtx_fine, partition, n_halo=1, min_length=1):
    new_partition = partition[:]
    part_len = len(partition)
    id_mask = [True]*part_len
    
    for part_id, idx in enumerate(partition):
        diff = idx[1] - idx[0]

        # TODO: set variable length
        if diff == 1:  # one element, a[i:i+1]
            # condition when merged with upper neighbor (if existing)
            if part_id == 0:
                # can only merge into lower partition
                i_lower_begin, i_lower_end = partition[part_id+1]
                new_partition[part_id+1] = i_lower_begin-1, i_lower_end

            elif part_id == part_len-1:
                # can only merge into upper partition
                i_upper_begin, i_upper_end = partition[part_id-1]
                new_partition[part_id-1] = i_upper_begin, i_upper_end+1     

            else:
                # decide on merge target depending on condition
                # XXX: should halo be passed here?
                i_lower_begin, i_lower_end = partition[part_id+1]
                i_upper_begin, i_upper_end = partition[part_id-1]
                
                cond_merge_with_lower = tridiag.tridiag_cond_partition(
                        mtx_fine, i_lower_begin-1, i_lower_end, n_halo)
                cond_merge_with_upper = tridiag.tridiag_cond_partition(
                        mtx_fine, i_upper_begin, i_upper_end+1, n_halo)
                
                if cond_merge_with_lower < cond_merge_with_upper:
                    new_partition[part_id+1] = i_lower_begin-1, i_lower_end
                else:
                    new_partition[part_id-1] = i_upper_begin, i_upper_end+1

            id_mask[part_id] = False

        elif diff == 0:
            id_mask[part_id] = False # no elements, a[i:i]
    
    # Remove elements that are masked from array
    new_partition = [new_partition[i] for i in range(0, len(id_mask)) if id_mask[i] is True]
    return new_partition, id_mask


def generate_partition(a_fine, b_fine, c_fine, M, n_halo, k_max_up, k_max_down):
    N_fine = len(a_fine)
    N_coarse = (ceil(N_fine / M)) * 2

    # Compute condition of fine system (only for numerical experiments)
    mtx_fine = rpta.bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    mtx_cond = np.linalg.cond(mtx_fine)
    # print("Fine system, condition: {:e}".format(mtx_cond))

    # Start with partition with blocks of fixed size
    static_partition = partition.generate_static_partition(N_fine, M)

    dyn_partition, mtx_cond_partmax, mtx_cond_partmax_dyn = tridiag.tridiag_dynamic_partition(
        mtx_fine, static_partition, n_halo, k_max_up, k_max_down)
    assert(N_coarse == len(dyn_partition)*2)
        
    # Merge partitions with only a single element
    # If multiple neighbors are available, choose based on minimal condition
    dyn_partition, id_mask = merge_partition(mtx_fine, dyn_partition, n_halo)
    if not all(id_mask):
        print('warning: changed partition size from {} to {}'.format(
            N_coarse / 2, len(dyn_partition)), file=sys.stderr)
    N_coarse = len(dyn_partition)*2

    return dyn_partition, mtx_cond, mtx_cond_partmax, mtx_cond_partmax_dyn


# XXX: "it must be true that `num_partitions_per_block` <= block dimension"
# TODO: use reduce_and_solve (rpta_util)
def rptapp_part_cond(a_fine, b_fine, c_fine, d_fine, N_tilde, M,
                     k_max_up=0, k_max_down=0, threshold=0, n_halo=1, level=0):
    # N_fine = len(a_fine)
    # N_coarse = (ceil(N_fine / M)) * 2

    dyn_partition, mtx_cond, mtx_cond_partmax, mtx_cond_partmax_dyn = generate_partition(
        a_fine, b_fine, c_fine, M, n_halo, k_max_up, k_max_down)
    N_coarse = len(dyn_partition)*2

    # Reduce to coarse system
    a_coarse = np.zeros(N_coarse)
    b_coarse = np.zeros(N_coarse)
    c_coarse = np.zeros(N_coarse)
    d_coarse = np.zeros(N_coarse)
    
    # TODO: document input/output parameters of rptapp_reduce()
    rpta.rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse, d_coarse, 
                       dyn_partition, threshold)
    mtx_coarse = rpta.bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)
    mtx_cond_coarse = np.linalg.cond(mtx_coarse)
    
    # print("\nCoarse system (M = {}, k_max_up = {}, k_max_down = {}), condition: {:e}".format(
    #     M, k_max_up, k_max_down, mtx_cond_coarse))

    # If system size below threshold, solve directly; otherwise, call recursively.
    # Note: each recursive call starts from a static partition, and computes
    # new boundaries based on condition of the partitions
    # TODO: set maximum level for adjusting partition boundaries?
    if len(a_coarse) <= N_tilde:
        x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
    else:
        x_coarse = rptapp_part_cond(a_coarse, b_coarse, c_coarse, d_coarse, N_tilde, M, 
                                    k_max_up, k_max_down, threshold, n_halo, level=level+1)    

    # Substitute into fine system
    x_fine_rptapp = rpta.rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, 
                                           dyn_partition, threshold)
    # Return additional values for parameter study
    return x_fine_rptapp, mtx_cond, mtx_cond_coarse, mtx_cond_partmax, mtx_cond_partmax_dyn
  

# TODO: use reduce_and_solve (rpta_util)
def rptapp_part_cond_print(a_fine, b_fine, c_fine, d_fine, x_fine, mtx_id, N_tilde, M, 
                 k_max_up, k_max_down, threshold, n_halo):
    try:
        x_fine_rptapp, cond, cond_coarse, cond_partmax, cond_partmax_dyn = rptapp_part_cond(
                a_fine, b_fine, c_fine, d_fine, N_tilde, M, k_max_up, k_max_down, 0, n_halo)
        fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
        print("{},{},{},{},{},{},{},{},{}".format(
            mtx_id, M, k_max_up, k_max_down, fre, cond, cond_coarse, cond_partmax, cond_partmax_dyn))

    except np.linalg.LinAlgError:
        print("warning: Singular matrix detected", file=sys.stderr)
        print("{},{},{},{},{},{},{},{},{}".format(
            mtx_id, M, k_max_up, k_max_down, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf))


# TODO: take k_max_up, k_max_down, M as arguments
def main_part_cond(mtx_id, N_fine, n_halo):
    unif_low, unif_high = -1, 1
    print("ID,M,k_max_up,k_max_down,fre,cond,cond_coarse,cond_partmax,cond_partmax_dyn")

    np.random.seed(0)
    a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
        mtx_id, N_fine, unif_low, unif_high)

    # Take all combinations of partition size / k_max_up / k_max_down
    Ms = range(16, 65)
    k_sup = 6
    for M in Ms:
        #N_tilde = 64
        N_tilde = (ceil(N_fine / M)) * 2
    
        for k_max_up in range(0, k_sup):
            for k_max_down in range(0, k_sup):
                rptapp_part_cond_print(a_fine, b_fine, c_fine, d_fine, x_fine, mtx_id, N_tilde, M, 
                                       k_max_up, k_max_down, 0, n_halo)

if __name__ == "__main__":
    mtx_id = int(sys.argv[1])
    N_fine = int(sys.argv[2])
    n_halo = int(sys.argv[3])
    
    main_part_cond(mtx_id, N_fine, n_halo)