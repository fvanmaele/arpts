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
#import glob
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
    N_fine = 512
    n_samples = 1000
    M_range = range(32, 65)

    for id in [11, 14]:
        # Retrieve coefficient matrix
        mtx_data = []
        with open("mtx-{}-{}.json".format(id, N_fine)) as m_json:
            mtx_data = json.load(m_json)
        cond = mtx_data['condition']

        plt.hist([cond], bins=1)
        plt.xscale('log')
        plt.savefig("mtx-{}-{}-cond.png".format(id, N_fine), dpi=108)

        # Convert coefficient matrix into bands
        mtx_name = "mtx-{}-{}.mtx".format(id, N_fine)
        mtx = mmread(mtx_name)        
        a_fine_m, b_fine_m, c_fine_m = matrix.scipy_matrix_to_bands(mtx)

        # Retrieve (multiprecision) solution to linear system
        for rhs_i in range(1, 4):
            rhs_data = []
            with open("mtx-{}-{}-rhs{}.json".format(id, N_fine, rhs_i), 'r') as d_json:
                rhs_data = json.load(d_json)    

            rhs = rhs_data['rhs']
            x_fine_m = np.array(rhs_data['solution'])

            # Solve system with static partition (M: 32..64)
            fre_static = []

            for M in M_range:
                partition_static = partition.generate_static_partition(N_fine, M)
                x_rpta_static, mtx_coarse_static, mtx_cond_coarse_static = rpta.reduce_and_solve(
                    a_fine_m, b_fine_m, c_fine_m, rhs, partition_static, pivoting='scaled_partial')

                fre_static.append(np.linalg.norm(x_rpta_static - x_fine_m) / np.linalg.norm(x_fine_m))    
            
            # Iterate over n partitions (all for the same matrix)
            for n_holes in [8, 12, 16]:
                fre_dec = []

                # Load generated partitions from sample tests
                holes_data = []
                with open("holes-{}-16-80-{:0>2}.json".format(N_fine, n_holes)) as h_json:
                    holes_data = json.load(h_json)

                for i in range(0, n_samples):
                    print(mtx_name, ", cond: ", mtx_data['condition'], 
                          "rhs: ", rhs_i, " n_holes: ", n_holes, " sample: ", i)

                    # Convert holes (1-indexed) into partition (0-indexed)                   
                    holes_0idx = np.array(holes_data[i]) - 1
                    partition_decoupled = [[0, holes_0idx[0]]]
        
                    for prev, curr in zip(holes_0idx, holes_0idx[1:]):
                        partition_decoupled.append([prev, curr])
                    partition_decoupled.append([holes_0idx[-1], N_fine])
        
                    # Solve system with dynamic partition
                    x_rpta_dec, mtx_coarse_dec, mtx_cond_coarse_dec = rpta.reduce_and_solve(
                        a_fine_m, b_fine_m, c_fine_m, rhs, partition_decoupled, pivoting='scaled_partial')
                    fre_dec.append(np.linalg.norm(x_rpta_dec - x_fine_m) / np.linalg.norm(x_fine_m))

                # TODO: export results to .json?
                plot_errbar_rhs(fre_dec, fre_static, M_range, id, n_holes, n_samples, rhs_i, 1.0)


if __name__ == "__main__":
    # import argparse
    # parser = argparse.ArgumentParser(description='Retrieve arguments')
    # # XXX: add additional arguments
    # parser.add_argument("root", type=str, help="directory to be processed")

    # args = parser.parse_args()
    # process_directory(args.root)
    process_directory(os.getcwd())
