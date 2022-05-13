#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:18:23 2022

@author: ferdinand
"""
import argparse
import numpy as np
import matrix, rpta
import matplotlib.pyplot as plt

# from math import ceil
from scipy.io import mmread

from main_random import main_random
from main_cond_coarse import main_cond_coarse
from main_static import main_static
# from main_rows import main_rows


def setup(mtx_id, N_fine, mean=3, stddev=1):
    a_fine, b_fine, c_fine = matrix.scipy_matrix_to_bands(mmread("../mtx/{:02d}-{}".format(mtx_id, N_fine)))
    mtx = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)

    x_fine = np.random.normal(mean, stddev, N_fine)
    d_fine = np.matmul(mtx, x_fine)
    
    return a_fine, b_fine, c_fine, x_fine, d_fine, mtx


def generate_min_partition(generator, extractor):
    min_fre, min_fre_part = np.Inf, []
    min_cond, min_cond_part = np.Inf, []
    
    for sample in generator:
        _sol, _fre, _coarse, _cond_coarse, _part = extractor(sample)
        
        if _fre < min_fre:
            min_fre = _fre
            min_fre_part = _part
            
        if _cond_coarse < min_cond:
            min_cond = _cond_coarse
            min_cond_part = _part

    return min_fre, min_fre_part, min_cond, min_cond_part


def generate_test_case(test_case, a_fine, b_fine, c_fine, d_fine, x_fine, M=32):
    N_fine = len(a_fine)
    if test_case == 'random':
        generator = main_random(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                                1000, 32, 100)
        extractor = lambda sample : sample[0:5]
    
    elif test_case == 'reduce':
        generator = main_cond_coarse(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                                     list(range(16,41)), list(range(22,73)), 6)
        extractor = lambda sample : [sample[0]] + list(sample[3:])
    
    elif test_case == 'static':
        generator = main_static(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, 
                                list(range(16,65)), 6)
        extractor = lambda sample : [sample[0]] + list(sample[2:])

    else:
        raise ValueError('unknown test case')

    return generate_min_partition(generator, extractor)    


def run_trials(mtx, a_fine, b_fine, c_fine, part, label, mean=3, stddev=1, n_trials=5000):
    trials = [None]*n_trials
    N_fine = len(a_fine)
    N_coarse = len(part)*2

    for k in range(0, n_trials):
        x_fine_new = np.random.normal(mean, stddev, N_fine)
        d_fine_new = np.matmul(mtx, x_fine_new)
        
        # Solve linear system with new right-hand side
        x_fine_rptapp_new, mtx_coarse_new, mtx_cond_coarse_new = rpta.reduce_and_solve(
            N_coarse, a_fine, b_fine, c_fine, d_fine_new, part, threshold=0)
        
        if k % 20 == 0:
            print("trial #{}, {}".format(k, label))
        trials[k] = np.linalg.norm(x_fine_rptapp_new - x_fine_new) / np.linalg.norm(x_fine_new)

    return trials


def ecdf(a):
    x, counts = np.unique(a, return_counts=True)
    cumsum = np.cumsum(counts)

    return x, cumsum / cumsum[-1]


def plot_ecdf(v, names, filename=None):
    assert(len(v) == len(names))
    plt.clf()

    for i, a in enumerate(v):
        x, y = ecdf(a)
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0.)
        plt.plot(x, y, drawstyle='steps-post', label=names[i])

    plt.legend()
    plt.grid(True)

    if filename is not None:
        plt.savefig(filename)

# def P(trial, lt):
#     arr = ecdf(trial)
#     return arr[0][arr[1] <= lt]

# TODO: include additional arguments for generate_test_case()
def main():
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("M", type=int)
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    parser.add_argument("--n-trials", type=int, default=5000, help="amount of trials for generated partition")
    parser.add_argument("--mean", type=float, default=3, help="mean of generated solutions (normal distribution)")
    parser.add_argument("--stddev", type=float, default=1, help="standard deviation of generated solutions (normal distribution")
    args = parser.parse_args()
    np.random.seed(args.seed)   
    plt.rcParams["figure.figsize"] = (12,8)
    
    # Set up linear system
    a_fine, b_fine, c_fine, x_fine, d_fine, mtx = setup(args.mtx_id, args.N_fine, args.mean, args.stddev)

    # Generate dynamic partition from one sample
    min_fre_rand, min_fre_rand_part, min_cond_rand, min_cond_rand_part = generate_test_case(
        'random', a_fine, b_fine, c_fine, d_fine, x_fine)
    min_fre_reduce, min_fre_reduce_part, min_cond_reduce, min_cond_reduce_part = generate_test_case(
        'reduce', a_fine, b_fine, c_fine, d_fine, x_fine)
    min_fre_static, min_fre_static_part, min_cond_static, min_cond_static_part = generate_test_case(
        'static', a_fine, b_fine, c_fine, d_fine, x_fine)
    
    # Comparison to static partition with fixed M
    # FIXME: ValueError: not enough values to unpack (expected 6, got 1)
    _, _, static_fre, _, _, static_part = main_static(
        len(a_fine), a_fine, b_fine, c_fine, d_fine, x_fine, [args.M], 6)

    # Verify generated partition on a set of samples (x, from same distribution)
    trials_rand_fre = run_trials(
        mtx, a_fine, b_fine, c_fine, min_fre_rand_part, "random + fre", args.mean, args.stddev, args.n_trials)
    trials_rand_cond = run_trials(
        mtx, a_fine, b_fine, c_fine, min_cond_rand_part, "random + cond", args.mean, args.stddev, args.n_trials)
    trials_reduce_fre = run_trials(
        mtx, a_fine, b_fine, c_fine, min_fre_reduce_part, "reduce + fre", args.mean, args.stddev, args.n_trials)
    trials_reduce_cond = run_trials(
        mtx, a_fine, b_fine, c_fine, min_cond_reduce_part, "reduce + cond", args.mean, args.stddev, args.n_trials)
    trials_static_fre = run_trials(
        mtx, a_fine, b_fine, c_fine, min_fre_static_part, "static + fre", args.mean, args.stddev, args.n_trials)
    trials_static_cond = run_trials(
        mtx, a_fine, b_fine, c_fine, min_cond_static_part, "static + cond", args.mean, args.stddev, args.n_trials)
    trials_static = run_trials(
        mtx, a_fine, b_fine, c_fine, static_part, "static", args.mean, args.stddev, args.n_trials)

    # Plot empirical cumulative distribution
    trials_v = [trials_rand_fre, trials_rand_cond, trials_reduce_fre, trials_reduce_cond, 
                trials_static_fre, trials_static_cond, trials_static]
    trials_n = ['rand_min_fre', 'rand_min_cond', 'reduce_min_fre', 'reduce_min_cond', 
                'static_min_fre', 'static_min_cond', 'static_M{:2d}'.format(args.M)]
    filename = "ecdf_{:02d}".format(args.mtx_id)
    plot_ecdf(trials_v, trials_n, filename)


if __name__ == "__main__":
    main()