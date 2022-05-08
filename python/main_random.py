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


def main_setup(mtx_id, N_fine):
    a_fine, b_fine, c_fine = matrix.scipy_matrix_to_bands(
        mmread("../mtx/{:02d}-{}".format(mtx_id, N_fine)))

    # Solution
    mtx = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    x_fine = np.random.normal(3, 1, N_fine)

    # Right-hand side
    d_fine = np.matmul(mtx, x_fine)
    
    return a_fine, b_fine, c_fine, d_fine, x_fine


def main_random(mtx_id, N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
                part_min, part_max):
    rpta_partition = partition.generate_random_partition(N_fine, part_min, part_max)
    N_coarse = len(rpta_partition)*2

    # Main computation step
    x_fine_rptapp, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
        N_coarse, a_fine, b_fine, c_fine, d_fine, rpta_partition, threshold=0)

    if x_fine_rptapp is not None:
        fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
    else:
        fre = np.Inf
        
    print('{},{},{:e},{:e}'.format(mtx_id, N_fine, fre, mtx_cond_coarse))
    return x_fine_rptapp, fre, mtx_coarse, mtx_cond_coarse, rpta_partition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("n_samples", type=int)
    parser.add_argument("part_min", type=int)
    parser.add_argument("part_max", type=int)
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Generate tridiagonal system
    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(args.mtx_id, args.N_fine)
    
    # Solve it with randomly chosen partitions
    for n in range(0, args.n_samples):
        main_random(args.mtx_id, args.N_fine, a_fine, b_fine, c_fine, d_fine, x_fine,
                    args.part_min, args.part_max)
