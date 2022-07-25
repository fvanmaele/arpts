#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:02:37 2022

@author: Ferdinand Vanmaele
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from scipy.stats import truncnorm

import matplotlib # __version__
from packaging import version


def generate_static_partition(N_fine, M, threshold=1):
    """
    Generate a partition of the index set [0, 1, ..., N_fine-1] where each partition has size M
    (if N divides M evenly), or the last partition has size N % M.  If N % M is below a certain 
    threshold, then the last partition is merged to its upper neighbor.

    Parameters
    ----------
    N_fine : int
        Size of the index set.
    M : int
        Size of the partition.
    threshold : int
        Minimum partition size.

    Returns
    -------
    partition_idx : list
        Nested list where each item [i_begin, i_end] is the half-open interval
        denoting the beginning and end of the partition.

    """
    partition_idx = []

    if N_fine // M == int(N_fine / M):
        n_partitions = (N_fine // M)
    else:
        n_partitions = (N_fine // M) + 1

    for i in range(0, N_fine//M):
        partition_idx.append([i*M, (i+1)*M])
    
    # Compute remaining partition of size N % M
    N_mod = N_fine % M

    if N_mod > 0 and N_mod <= threshold:
        partition_idx[-1] = [i*M, N_fine]
    elif N_mod > 0:
        partition_idx.append([n_partitions*M, N_fine])

    return partition_idx


def generate_partition_func(a_fine, b_fine, c_fine, part_min, part_max, 
                                   func, argopt):
    N = len(a_fine)
    assert(part_min < part_max)
    assert(part_min > 0)
    assert(part_max < N-1)

    partition = []
    i_begin = 0 # Index of first row (upper boundary)    
    while i_begin + part_max < N:
        f_values = []
        for offset in range(part_min, part_max):
            i_target = min(i_begin + offset, N-1)
            mtx = np.matrix([[b_fine[i_begin], c_fine[i_begin]], 
                             [a_fine[i_target], b_fine[i_target]]])
            f_values.append(abs(func(mtx)))
            # print("{}, {}: |det| = {}".format(
            #     i_begin, i_target, abs(np.linalg.det(mtx))))

        # Criterion: maximum determinant (or minimum condition)
        f_values_argopt = argopt(f_values)
        # print("{}, {}: |det| (max) = {}".format(
        #     i_begin, i_begin + part_min + f_values_argopt, f_values[f_values_argopt]))
        partition.append([i_begin, i_begin + part_min + f_values_argopt])

        # Go to next partition
        i_begin = min(i_begin + part_min + f_values_argopt, N)
    
    # Append last partition
    if i_begin < N:
        # mtx = np.matrix([[b_fine[i_begin], c_fine[i_begin]], 
        #                  [a_fine[N-1], b_fine[N-1]]])
        # f_value = abs(func(mtx))
        # print("{}, {}: |det| = {}".format(i_begin, N-1, f_value))
        partition.append([i_begin, N])

    return partition


def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def randint_normal(size, mean=0, sd=1, low=0, upp=10):
    return get_truncated_normal(mean, sd, low, upp).rvs(size).round().astype(int)


def generate_random_partition_normal(N, part_min, part_max, part_mean, part_sd):
    assert(part_min < part_max)
    assert(part_min > 0)
    assert(part_max < N-1)

    partition = []
    partition_begin = 0
    remainder = N

    while remainder >= 0:
        offset = randint_normal(1, mean=part_mean, sd=part_sd, low=part_min, upp=part_max)[0]
        partition.append([partition_begin, min(partition_begin+offset, N)])
        partition_begin += offset
        remainder -= offset

    # If last partition is too low, merge into upper neighbor
    if partition[-1][1] - partition[-1][0] < part_min:
        partition[-2] = [partition[-2][0], N]
        partition.pop()
    
    return partition


def generate_random_partition(N, part_min, part_max):
    assert(part_min < part_max)
    assert(part_min > 0)
    assert(part_max < N-1)

    partition = []
    partition_begin = 0
    remainder = N

    while remainder >= 0:
        offset = np.random.randint(part_min, part_max)
        partition.append([partition_begin, min(partition_begin+offset, N)])
        partition_begin += offset
        remainder -= offset

    # If last partition is too low, merge into upper neighbor
    if partition[-1][1] - partition[-1][0] < part_min:
        partition[-2] = [partition[-2][0], N]
        partition.pop()
    
    return partition


def plot_coarse_system(mtx_coarse, title='Coarse system', cmap='bwr'):
    """
    Visualization of a low-dimensional matrix. If the matrix has negative
    values, choose 0 as the center of the color map (white); otherwise,
    choose the average of maximum and minimum value.

    Parameters
    ----------
    mtx_coarse : np.ndarray
        Matrix array (2-dimensional)
    title : str, optional
        Title of the plot. The default is 'Coarse system'.
    cmap : str, optional
        Color map used for the matrix. The default is 'bwr'.

    Returns
    -------
    None.

    """
    vmin=np.min(mtx_coarse)
    vmax=np.max(mtx_coarse)
    vcenter = 0 if vmin < 0 else (vmin+vmax)/2

    # DivergingNorm in matplotlib 3.1, TwoSlopeNorm in 3.2    
    if version.parse(matplotlib.__version__) < version.parse("3.2"):
        norm_coarse = colors.DivergingNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
        norm_coarse = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plt.matshow(mtx_coarse, cmap=cmap, norm=norm_coarse)
    plt.title(title)
    plt.colorbar()
    plt.show()


def main():
    print("testing static partition, n = 512, m = 32")
    n = 512
    m_even = 32
    part_m_even = generate_static_partition(n, m_even)
    print(part_m_even)
    
    # Test if intervals are half-open
    for idx in range(1, len(part_m_even)):
        assert(part_m_even[idx][0] == part_m_even[idx-1][1])
    assert(len(part_m_even) == 16)
    assert(part_m_even[15][1] == n)
    
    # Case where M does not divide N (remainder N % M)
    print("testing static partition, n = 512, m = 33")
    m_odd = 33
    part_m_odd = generate_static_partition(n, m_odd)
    print(part_m_odd)
    
    for idx in range(1, len(part_m_even)):
        assert(part_m_even[idx][0] == part_m_even[idx-1][1])
    assert(len(part_m_even) == 16)
    assert(part_m_even[15][1] == n)

    print("tests OK")

if __name__ == "__main__":
    main()