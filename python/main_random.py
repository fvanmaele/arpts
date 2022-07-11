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


def main_random(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, n_samples, distribution, part_min, part_max, pivoting,
                part_mean=None, part_sd=None):
    if distribution == 'normal':
        assert(part_mean is not None)
        assert(part_sd is not None)
        gen = lambda N, pmin, pmax, pmean, psd : partition.generate_random_partition_normal(N, pmin, pmax, pmean, psd)
    
    elif distribution == 'uniform':
        gen = lambda N, pmin, pmax, pmean, psd : partition.generate_random_partition(N, pmin, pmax)

    else:
        raise ValueError("--distribution must be 'normal' or 'uniform")

    for n in range(0, n_samples):
        rpta_partition = gen(N_fine, part_min, part_max, part_mean, part_sd)
        N_coarse = len(rpta_partition)*2

        # Main computation step
        x_fine_rptapp, mtx_coarse, mtx_cond_coarse = rpta.reduce_and_solve(
            N_coarse, a_fine, b_fine, c_fine, d_fine, rpta_partition, pivoting=pivoting)

        if x_fine_rptapp is not None:
            fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
        else:
            fre = np.Inf

        yield x_fine_rptapp, fre, mtx_coarse, mtx_cond_coarse, rpta_partition


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("n_samples", type=int)
    parser.add_argument("part_min", type=int)
    parser.add_argument("part_max", type=int)
    parser.add_argument("--distribution", type=str, default='uniform', help="distribution of partition boundaries")
    parser.add_argument("--part-mean", type=float, help="mean for --distribution=normal")
    parser.add_argument("--part-sd", type=float, help="standard deviation for --distribution=normal")
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    parser.add_argument("--pivoting", type=str, default='scaled_partial', help="type of pivoting used")
    args = parser.parse_args()
    np.random.seed(args.seed)

    # Generate tridiagonal system
    a_fine, b_fine, c_fine, d_fine, x_fine = main_setup(args.mtx_id, args.N_fine)
    
    # Solve it with randomly chosen partitions
    for sample in main_random(args.N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                              args.n_samples, args.distribution, args.part_min, 
                              args.part_max, args.pivoting, args.part_mean, args.part_sd):
        x_fine_rptapp, fre, mtx_coarse, mtx_cond_coarse, rpta_partition = sample
        print('{},{},{:e},{:e}'.format(args.mtx_id, args.N_fine, fre, mtx_cond_coarse))
