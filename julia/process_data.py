#!/usr/bin/python3
import os
import sys

home_dir = os.environ['HOME']
python_dir = '{}/source/repos/arpts/python'.format(home_dir)
sys.path.append(python_dir)

import rpta, matrix, partition
import numpy as np
from scipy.io import mmread

import json
import glob
import matplotlib.pyplot as plt


def plot_errbar_rhs(fre_dec, fre_static, M_range, mtx_id, n_holes, n_samples, rhs_i, eps):
    fre_idx = [0]
    fre_lab = ["D"]
    fre_mean = [np.mean(fre_dec)]
    fre_std = [np.std(fre_dec)]
    fre_static_t = np.array(fre_static).T

    for k, M in enumerate(M_range):
        fre_mean.append(np.mean(fre_static_t[k]))
        fre_std.append(np.std(fre_static_t[k]))
        fre_idx.append(k+1)
        fre_lab.append(str(M))

    plt.figure(figsize=(10, 4))
    plt.title("mtx_id {} - n_holes {} - n_samples {} - rhs {} - eps {:.1e}".format(mtx_id, n_holes, n_samples, rhs_i, eps))
    plt.xticks(ticks=fre_idx, labels=fre_lab)

    plt.yscale('log')
    plt.errorbar(fre_idx, fre_mean, fre_std, linestyle='None', marker='o', capsize=3)
    plt.tight_layout()
    plt.savefig("mtx-{}-{:0>2}-{}-rhs{}-eps{:.1e}.png".format(mtx_id, n_holes, n_samples, rhs_i, eps), dpi=108)


def process_directory(root):
    # List of all matrix samples
    mtx_decoupled = glob.glob(os.path.join(root, "*.mtx"))
    mtx_decoupled.sort()

    # Retrieve matrix information
    # XXX: redundant information for eps=1.0
    mtx_data = []
    for m in mtx_decoupled:
        m_name, m_ext = os.path.splitext(m)
        # XXX: m_name + ".json" (matrix metadata)
        with open(m_name + ".json", 'r') as m_json:
            mtx_data.append(json.load(m_json))

    # XXX: flat arrays mtx_data['mtx_id']
    n_samples = len(mtx_decoupled)
    conds   = np.array([d['condition'] for d in mtx_data])
    mtx_id  = mtx_data[0]['mtx_id']
    n_holes = mtx_data[0]['n_holes']
    eps     = mtx_data[0]['eps']
    N_fine  = mtx_data[0]['N']
    M_range = range(32, 65)  # XXX: possible command-line argument

    # Histogram of matrix conditions (logarithmic scale)
    hist, bins, _ = plt.hist(conds, bins=50)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.clf()
    plt.hist(conds, bins=logbins)
    plt.xscale('log')
    plt.savefig("mtx-{}-{:0>2}-{}-eps{:.1e}-cond.png".format(mtx_id, n_holes, n_samples, eps), dpi=108)

    # Take quantiles of samples in cases of (strongly) varying condition numbers
    # (stddev as measure of numerical stability of algorithm)
    # XXX: possible command-line argument
    n_max_samples, qnt_l, qnt_r = 100, 0.4, 0.6
    conds_qnt = np.quantile(conds, np.linspace(qnt_l, qnt_r, n_max_samples), method='nearest')
    idx_qnt = np.argwhere(np.in1d(conds, conds_qnt)).flatten()
    assert(len(idx_qnt) == len(np.unique(idx_qnt)))

    # Retrieve (multiprecision) solution to linear system
    # XXX: fixed number (3) of right-hand sides
    for rhs_i in range(1, 4):
        glob_suffix = "*-rhs{}.json".format(rhs_i)
        rhs_decoupled = glob.glob(os.path.join(args.root, glob_suffix))
        rhs_decoupled.sort()

        rhs_data = []
        for d in rhs_decoupled:
            with open(d, 'r') as d_json:
                rhs_data.append(json.load(d_json))
        assert(len(rhs_data) == len(mtx_data))

        # Compute average FRE over 100 samples of max condition (10 outliers removed)
        # XXX: fixed right-hand side for all samples
        rhs = rhs_data[0]['rhs']
        fre_dec = []
        fre_static = []

        # Iterate over samples of highest condition
        for i in idx_qnt:
            # Convert coefficient matrix into bands
            mtx = mmread(mtx_decoupled[i])
            print(mtx_decoupled[i], ", cond: ", mtx_data[i]['condition'])

            a_fine_m, b_fine_m, c_fine_m = matrix.scipy_matrix_to_bands(mtx)
            x_fine_m = np.array(rhs_data[i]['solution'])

            # Convert holes (1-indexed) into partition (0-indexed)
            holes_0idx = np.array(mtx_data[i]['sample_1idx']) - 1
            partition_decoupled = [[0, holes_0idx[0]]]

            for prev, curr in zip(holes_0idx, holes_0idx[1:]):
                partition_decoupled.append([prev, curr])
            partition_decoupled.append([holes_0idx[-1], N_fine])

            # Solve system with dynamic partition
            x_rpta_dec, mtx_coarse_dec, mtx_cond_coarse_dec = rpta.reduce_and_solve(
                a_fine_m, b_fine_m, c_fine_m, rhs, partition_decoupled, pivoting='scaled_partial')
            fre_dec.append(np.linalg.norm(x_rpta_dec - x_fine_m) / np.linalg.norm(x_fine_m))
            
            # Solve system with static partition (M: 32..64)
            fre_static_i = []
            for M in M_range:
                partition_static = partition.generate_static_partition(N_fine, M)
                x_rpta_static, mtx_coarse_static, mtx_cond_coarse_static = rpta.reduce_and_solve(
                    a_fine_m, b_fine_m, c_fine_m, rhs, partition_static, pivoting='scaled_partial')

                fre_static_i.append(np.linalg.norm(x_rpta_static - x_fine_m) / np.linalg.norm(x_fine_m))    
            
            fre_static.append(fre_static_i)

        # TODO: export results to .json?
        plot_errbar_rhs(fre_dec, fre_static, M_range, mtx_id, n_holes, n_samples, rhs_i, eps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Retrieve arguments')
    # XXX: add additional arguments
    parser.add_argument("root", type=str, help="directory to be processed")

    args = parser.parse_args()
    process_directory(args.root)
