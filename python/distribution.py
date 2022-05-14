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
import json

# from math import ceil
from sys import stderr
from scipy.io import mmread

from main_random import main_random
from main_cond_coarse import main_cond_coarse
from main_static import main_static
# from main_rows import main_rows


def setup(mtx_id, N_fine, gen_x_fine):
    a_fine, b_fine, c_fine = matrix.scipy_matrix_to_bands(mmread("../mtx/{:02d}-{}".format(mtx_id, N_fine)))
    mtx = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)

    x_fine = gen_x_fine(N_fine)
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


def generate_test_case(test_case, a_fine, b_fine, c_fine, d_fine, x_fine, *main_args):
    N_fine = len(a_fine)
    if test_case == 'random':
        generator = main_random(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, *main_args)
        extractor = lambda sample : sample[0:5]
    
    elif test_case == 'reduce':
        generator = main_cond_coarse(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, *main_args)
        extractor = lambda sample : [sample[0]] + list(sample[3:])
    
    elif test_case == 'static':
        generator = main_static(N_fine, a_fine, b_fine, c_fine, d_fine, x_fine, *main_args)
        extractor = lambda sample : [sample[0]] + list(sample[2:])

    else:
        raise ValueError('unknown test case')

    return generate_min_partition(generator, extractor)    


# TODO: allow to choose the distribution of the generated solutions
def run_trials(mtx, a_fine, b_fine, c_fine, part, label, gen_samples, n_trials=5000):
    trials = [None]*n_trials
    N_fine = len(a_fine)
    N_coarse = len(part)*2

    for k in range(0, n_trials):
        x_fine_new = gen_samples(N_fine)
        d_fine_new = np.matmul(mtx, x_fine_new)
        
        # Solve linear system with new right-hand side
        x_fine_rptapp_new, mtx_coarse_new, mtx_cond_coarse_new = rpta.reduce_and_solve(
            N_coarse, a_fine, b_fine, c_fine, d_fine_new, part, threshold=0)
        
        if k % 20 == 0:
            print("trial #{}, {}".format(k, label))
        trials[k] = np.linalg.norm(x_fine_rptapp_new - x_fine_new) / np.linalg.norm(x_fine_new)

    return trials


def ecdf(a):
    x, counts = np.unique(a, return_counts=True) # returns sorted unique elements
    cumsum = np.cumsum(counts)

    return x, cumsum / cumsum[-1]


def plot_ecdf(d, filename=None):
    plt.clf()

    for label, a in d.items():
        x, y = ecdf(a)
        x = np.insert(x, 0, x[0])
        y = np.insert(y, 0, 0.)
        plt.plot(x, y, drawstyle='steps-post', label=label)

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
    # positional arguments
    parser.add_argument("mtx_id", type=int)
    parser.add_argument("N_fine", type=int)
    parser.add_argument("M", type=int, help="fixed block size for comparison purposes")
    # global options
    parser.add_argument("--seed", type=int, default=0, help="value for np.random.seed()")
    parser.add_argument("--n-trials", type=int, default=5000, help="amount of trials for generated partition")
    parser.add_argument("--mean", type=float, default=3, help="mean of generated solutions (--distribution normal)")
    parser.add_argument("--stddev", type=float, default=1, help="standard deviation of generated solutions (--distribution normal)")
    parser.add_argument("--low", type=float, default=0, help="lower boundary of generated solutions (--distribution uniform)")
    parser.add_argument("--high", type=float, default=1, help="upper boundary of generated solutions (--distribution uniform)")
    parser.add_argument("--distribution", type=str, default='normal', help="distribution of generated solutions ('normal' or 'uniform')")
    parser.add_argument("--distribution-setup", type=str, default='normal', help="distribution for solution of generated partition")
    # options for random partition
    parser.add_argument("--rand-n-samples", type=int, default=1000, help="amount of samples for randomly generated partitions")
    parser.add_argument("--rand-min-part", type=int, default=32, help="minimal size of randomly generated partitions")
    parser.add_argument("--rand-max-part", type=int, default=100, help="maximal size of randomly generated partitions")
    # options for static partition
    parser.add_argument("--static-M-min", type=int, default=16, help="minimal block size for fixed partitions")
    parser.add_argument("--static-M-max", type=int, default=64, help="maximal block size for fixed partitions")
    # TODO: options for partition generated during reduction

    args = parser.parse_args()
    np.random.seed(args.seed)
    plt.rcParams["figure.figsize"] = (12,8)
    
    # Define distribution of generated solutions
    if args.distribution == "normal":
        gen_samples = lambda N : np.random.normal(args.mean, args.stddev, N)
        print("[SOL] Normal distribution with mean {}, standard deviation {}".format(args.mean, args.stddev), file=stderr)
    elif args.distribution == "uniform":
        gen_samples = lambda N : np.random.uniform(args.low, args.high, N)
        print("[SOL] Uniform distribution with values lower bound {}, upper bound {}".format(args.low, args.high))
    else:
        raise ValueError("invalid distribution specified (--distribution)")
    
    # Define distribution of sample used to generate partition
    if args.distribution_setup == "normal":
        gen_samples_part = lambda N : np.random.normal(args.mean, args.stddev, N)
        print("[PART] Normal distribution with mean {}, standard deviation {}".format(args.mean, args.stddev), file=stderr)
    elif args.distribution_setup == "uniform":
        gen_samples_part = lambda N : np.random.uniform(args.low, args.high, N)
        print("[PART] Uniform distribution with values lower bound {}, upper bound {}".format(args.low, args.high))
    else:
        raise ValueError("invalid distribution specified (--distribution-setup)")
    
    # Set up linear system
    a_fine, b_fine, c_fine, x_fine, d_fine, mtx = setup(args.mtx_id, args.N_fine, gen_samples_part)

    # Comparison to static partition with fixed M
    static = [S for S in main_static(
        len(a_fine), a_fine, b_fine, c_fine, d_fine, x_fine, [args.M], 6)]
    static_fre, static_part = static[0][2], static[0][5]

    # Generate dynamic partition from one sample
    min_fre_rand, min_fre_rand_part, min_cond_rand, min_cond_rand_part = generate_test_case(
        'random', a_fine, b_fine, c_fine, d_fine, x_fine, 
        args.rand_n_samples, args.rand_min_part, args.rand_max_part)
    min_fre_reduce, min_fre_reduce_part, min_cond_reduce, min_cond_reduce_part = generate_test_case(
        'reduce', a_fine, b_fine, c_fine, d_fine, x_fine, 
        list(range(16,41)), list(range(22,73)), 6)
    min_fre_static, min_fre_static_part, min_cond_static, min_cond_static_part = generate_test_case(
        'static', a_fine, b_fine, c_fine, d_fine, x_fine, 
        list(range(args.static_M_min,args.static_M_max+1)), 6)

    # Verify generated partition on a set of samples (x, from same distribution)
    trials_static = run_trials(
        mtx, a_fine, b_fine, c_fine, static_part, "static", gen_samples, args.n_trials)
    trials_rand_fre = run_trials(
        mtx, a_fine, b_fine, c_fine, min_fre_rand_part, "random + fre", gen_samples, args.n_trials)
    trials_rand_cond = run_trials(
        mtx, a_fine, b_fine, c_fine, min_cond_rand_part, "random + cond", gen_samples, args.n_trials)
    trials_reduce_fre = run_trials(
        mtx, a_fine, b_fine, c_fine, min_fre_reduce_part, "reduce + fre", gen_samples, args.n_trials)
    trials_reduce_cond = run_trials(
        mtx, a_fine, b_fine, c_fine, min_cond_reduce_part, "reduce + cond", gen_samples, args.n_trials)
    trials_static_fre = run_trials(
        mtx, a_fine, b_fine, c_fine, min_fre_static_part, "static + fre", gen_samples, args.n_trials)
    trials_static_cond = run_trials(
        mtx, a_fine, b_fine, c_fine, min_cond_static_part, "static + cond", gen_samples, args.n_trials)

    # Plot empirical cumulative distribution
    trials_d = {
        'rand_min_fre'    : trials_rand_fre,
        'rand_min_cond'   : trials_rand_cond,
        'reduce_min_fre'  : trials_reduce_fre,
        'reduce_min_cond' : trials_reduce_cond,
        'static_min_fre'  : trials_static_fre,
        'static_min_cond' : trials_static_cond,
        'static_M32'      : trials_static
    }

    # Use a common suffix which includes the distribution and matrix ID
    f_suffix = "{:02d}_{}_sol-{}_gen-{}".format(
        args.mtx_id, args.N_fine, args.distribution[0:4], args.distribution_setup[0:4])
    plot_ecdf(trials_d, "ecdf_{}".format(f_suffix))

    # Write trial data to file (JSON) for later use
    with open('trials_{}.json'.format(f_suffix), 'w') as outfile:
        json.dump(trials_d, outfile)


if __name__ == "__main__":
    main()