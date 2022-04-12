#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:02:37 2022

@author: Ferdinand Vanmaele
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def generate_static_partition(N_fine, M):
    """
    Generate a partition of the index set [0, 1, ..., N_fine-1] where each
    partition has size M.

    Parameters
    ----------
    N_fine : int
        Size of the index set.
    M : int
        Size of the partition.

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
    if N_fine % M > 0:
        partition_idx.append([n_partitions*M, N_fine])

    return partition_idx


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
    norm_coarse = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plt.matshow(mtx_coarse, cmap=cmap, norm=norm_coarse)
    plt.title(title)
    plt.colorbar()
    plt.show()


def tridiag_cond_partition(mtx_fine, i_begin, i_end, n_halo):
    """
    Compute the condition number for a partition of a tridiagonal matrix. 
    The condition is computed with np.linalg.cond, so this method should only
    be used for low-dimensional matrices.
    
    Parameters
    ----------
    mtx_fine : np.ndarray
        A view on the matrix of full dimension.
    i_begin : int
        Starting row and column index.
    i_end : int
        Ending row and column index (half-open).
    n_halo : int, optional
        Include additional row and columns. The default is 0.

    Returns
    -------
    mtx_part_cond : float
        Condition number of the partition.
    mtx_part_shape : list
        Size of the partition.

    """
    mtx_shape = np.shape(mtx_fine)
    N_fine = mtx_shape[0]
    assert(mtx_shape[1] == N_fine)

    # Take halo into account for partition
    i_new_begin = max(0, i_begin - n_halo)
    i_new_end = min(i_end + n_halo, N_fine)

    mtx_part = mtx_fine[i_new_begin:i_new_end, i_new_begin:i_new_end]
    mtx_part_cond = np.linalg.cond(mtx_part)
    #mtx_part_shape = np.shape(mtx_part)

    return mtx_part_cond, mtx_part


# XXX: Possible to use this for downshift by reversing views?
# Note: invariants on part (half-open intervals)
def tridiag_cond_upshift(mtx_fine, part, part_id, n_halo, k_max_up=5):
    if part_id == 0:
        return part # first partition, cannot extended boundaries upwards

    mtx_fine_shape = np.shape(mtx_fine)
    N_fine = mtx_fine_shape[0] # quadratic matrix

    conds_upshift = []
    conds_upshift_neigh = []
    new_part = part[:] # create copy by slicing

    i_begin, i_end = part[part_id] # boundaries of partition
    i_neigh_begin, i_neigh_end = part[part_id-1] # boundaries of upper neighbor

    # Extend boundaries of partition upwards
    # Note: by including 0, the condition for the original bounds is included.
    # This way the original partition is included when computing the minimum
    # (in case shifting boundaries results in a higher condition).
    for k_up in range(0, k_max_up+1):
        i_new_begin = i_begin - k_up        
        
        # Compute condition for partition
        mtx_cond, mtx = tridiag_cond_partition(
            mtx_fine, i_new_begin, i_end, n_halo)
        conds_upshift.append(mtx_cond)

        # Compute condition for upper neighbor with new boundary (half-open intervals)
        mtx_neigh_cond, mtx_neigh = tridiag_cond_partition(
            mtx_fine, i_neigh_begin, i_new_begin, n_halo)
        conds_upshift_neigh.append(mtx_neigh_cond)

        print("Partition {} (A_PP [upshift {}], size {}x{}), condition {:e}".format(
            part_id, k_up, np.shape(mtx)[0], np.shape(mtx)[1], mtx_cond))
        print("Partition {} (A_PP [upshift {}, neigh], size {}x{}), condition {:e}".format(
            part_id-1, k_up, np.shape(mtx_neigh)[0], np.shape(mtx_neigh)[1], mtx_neigh_cond))
        plot_coarse_system(mtx, "partition, k_up = {}, cond = {:e}".format(
            k_up, mtx_cond))

    # Compute minimal condition number for partition and (upper) neighbor
    conds_upshift_min_k = np.argmin(conds_upshift)
    conds_upshift_neigh_min_k = np.argmin(conds_upshift_neigh)

    print("argmin k_up = {}".format(conds_upshift_min_k))
    print("argmin k_up = {} [neigh]".format(conds_upshift_neigh_min_k))
    
    # Heuristic: if minimal neighbor has condition of a higher magnitude 
    # than the corresponding neighbor for the minimal partition, swap and check
    # XXX: this does not cover differences within the same order of magnitude
    # In some cases, an improvement in condition necessarily leads to a worse condition
    # of the neighbor (e.g. matrix 14), and we have to choose which to improve.
    # XXX: Recompute maxima in this case? (in rptapp)
    conds_new_k = conds_upshift_min_k
    if conds_upshift[conds_upshift_neigh_min_k] < 10*conds_upshift[conds_upshift_min_k]:
        if conds_upshift_neigh[conds_upshift_min_k] < 10*conds_upshift_neigh[conds_upshift_neigh_min_k]:
            conds_new_k = conds_upshift_neigh_min_k
    print("argmin k_up = {} [heuristic]".format(conds_new_k))

    # Return new bounds
    new_part[part_id] = [i_begin - conds_new_k, i_end]
    new_part[part_id-1] = [i_neigh_begin, i_begin - conds_new_k]
    return new_part, conds_upshift[conds_new_k], conds_upshift_neigh[conds_new_k]


def main():
    print("testing static partition, n = 512, m = 32")
    n = 512
    m_even = 32
    part_m_even = generate_static_partition(n, m_even)
    #print(part_m_even)
    
    # Test if intervals are half-open
    for idx in range(1, len(part_m_even)):
        assert(part_m_even[idx][0] == part_m_even[idx-1][1])
    assert(len(part_m_even) == 16)
    assert(part_m_even[15][1] == n)
    
    # Case where M does not divide N (remainder N % M)
    print("testing static partition, n = 512, m = 33")
    m_odd = 33
    part_m_odd = generate_static_partition(n, m_odd)
    #print(part_m_odd)
    
    for idx in range(1, len(part_m_even)):
        assert(part_m_even[idx][0] == part_m_even[idx-1][1])
    assert(len(part_m_even) == 16)
    assert(part_m_even[15][1] == n)

    print("tests OK")

if __name__ == "__main__":
    main()