#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:02:37 2022

@author: Ferdinand Vanmaele
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def generate_static_partition(N_fine, M, halo_n=0):
    """
    Generate a partition of the index set [0, 1, ..., N_fine-1] where each
    partition has size M. Optionally, a halo can be specified such that
    there are overlapping elements.

    Parameters
    ----------
    N_fine : int
        Size of the index set.
    M : int
        Size of the partition.
    halo_n : int, optional
        Size of the halo. The default is 0.

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
        if i > 0:
            # subtract 1 for nodes A_IP (upper halo)
            i_begin = i*M - halo_n
        else:
            i_begin = i*M
        
        if i == n_partitions-1:
            # we are in the last partition, of size M
            i_end = (i+1)*M # N_fine
        else:
            # we are in an inner partition, of size M
            i_end = (i+1)*M + halo_n
            
        partition_idx.append([i_begin, i_end])
    
    # Compute remaining partition of size N % M
    if N_fine % M > 0:
        partition_idx.append([n_partitions*M - halo_n, N_fine])

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


# XXX: Possible to use this for downshift by reversing views?
# Note: invariants on part (half-open intervals)
# TODO: implement halo for computing calculations (cannot be part of the partition,
# as it is used by the tridiagonal solver to solve the linear system)
def tridiag_cond_upshift(mtx_fine, part, part_num, k_max_up=5, n_halo=0):
    if part_num == 0:
        return part # first partition, cannot extended boundaries upwards

    conds_upshift = []
    conds_upshift_neigh = []
    new_part = part[:] # create copy by slicing

    i_begin, i_end = part[part_num] # boundaries of partition
    i_neigh_begin, i_neigh_end = part[part_num-1] # boundaries of upper neighbor

    # Extend boundaries of partition upwards
    for k_up in range(0, k_max_up+1):
        i_new_begin = i_begin - k_up

        # Compute condition for partition
        mtx_new = mtx_fine[i_new_begin:i_end, i_new_begin:i_end]
        mtx_new_cond = np.linalg.cond(mtx_new)
        mtx_new_shape = np.shape(mtx_new)
        conds_upshift.append(mtx_new_cond)

        print("Partition {} (A_PP [upshift {}], size {}x{}), condition {:e}".format(
            part_num, k_up, mtx_new_shape[0], mtx_new_shape[1], mtx_new_cond))

        i_neigh_end = i_new_begin

        # Compute condition for upper neighbor with new boundary (half-open intervals)        
        mtx_neigh = mtx_fine[i_neigh_begin:i_neigh_end, i_neigh_begin:i_neigh_end]
        mtx_neigh_cond = np.linalg.cond(mtx_neigh)
        mtx_neigh_shape = np.shape(mtx_neigh)
        conds_upshift_neigh.append(mtx_neigh_cond)
    
        print("Partition {} (A_PP [upshift {}, neigh], size {}x{}), condition {:e}".format(
            part_num-1, k_up, mtx_neigh_shape[0], mtx_neigh_shape[1], mtx_neigh_cond))

        plot_coarse_system(mtx_new, "partition, k_up = {}, cond = {:e}".format(
            k_up, mtx_new_cond))

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
    conds_upshift_final_min_k = conds_upshift_min_k
    if conds_upshift[conds_upshift_neigh_min_k] < 10*conds_upshift[conds_upshift_min_k]:
        if conds_upshift_neigh[conds_upshift_min_k] < 10*conds_upshift_neigh[conds_upshift_neigh_min_k]:
            conds_upshift_final_min_k = conds_upshift_neigh_min_k
    print("argmin k_up = {} [heuristic]".format(conds_upshift_final_min_k))

    # Return new bounds
    new_part[part_num] = [i_begin - conds_upshift_final_min_k, i_end]
    new_part[part_num-1] = [i_neigh_begin, i_begin - conds_upshift_final_min_k]
    return new_part


def main():
    print("testing static partition, n = 512, m = 32")
    n = 512
    m_even = 32
    part_m_even = generate_static_partition(n, m_even, 0)
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