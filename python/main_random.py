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

# TODO: take seed as argument for solution vector (fixed for generated matrix)
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


# TODO: return coarse system instead of condition (for further study)
def main_random(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
                n_samples, part_min, part_max, min_over='fre', seed=0):
    np.random.seed(seed)
    fre_v, cond_v, mtx_v = [], [], []

    for n in range(0, n_samples):
        rpta_partition = partition.generate_random_partition(
                N_fine, part_min, part_max)
        N_coarse = len(rpta_partition)*2
        
        fre, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
            N_coarse, a_fine, b_fine, c_fine, d_fine, x_fine, rpta_partition, threshold=0)
        # print("{},{},{:e},{:e}".format(mtx_id, n, fre, cond_coarse), file=sys.stderr)

        fre_v.append(fre)
        mtx_v.append(mtx_coarse)
        cond_v.append(mtx_cond_coarse)

    if min_over == "fre":
        min_idx = np.argmin(fre_v)

    elif min_over == "cond":
        min_idx = np.argmin(cond_v)

    print('{},{},{:e},{:e}'.format(mtx_id, min_idx, fre_v[min_idx], cond_v[min_idx]))
    return fre_v[min_idx], mtx_v[min_idx], cond_v[min_idx]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("n_samples", type=int)
    parser.add_argument("part_min", type=int)
    parser.add_argument("part_max", type=int)
    parser.add_argument("min_over")
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    args = parser.parse_args()

    if args.min_over != "fre" and args.min_over != "cond":
        raise ValueError

    # Generate tridiagonal system
    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(args.mtx_id, args.N_fine)
    
    # Solve it with randomly chosen partitions
    main_random(args.mtx_id, args.N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
                args.n_samples, args.part_min, args.part_max, args.min_over, args.seed)