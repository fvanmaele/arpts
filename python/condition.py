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
        View on the matrix of full dimension.
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
    mtx_part : np.ndarray
        View on the partition.

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
def tridiag_cond_shift(mtx_fine, part, part_id, n_halo, k_max_up=5, k_max_down=5):
    assert(k_max_down >= 0)
    assert(k_max_up >= 0)
    #mtx_fine_shape = np.shape(mtx_fine)
    #N_fine = mtx_fine_shape[0] # quadratic matrix

    conds_shift = {} # partition (indexed by tuples: k_up, k_down)
    conds_shift_upper = [] # upper neighbor
    conds_shift_lower = [] # lower neighbor
    new_part = part[:] # create copy by slicing

    # Assumptions on k_max_up and k_max_down:
    # 1. first partition cannot be extended upwards
    k_max_up = k_max_up if part_id > 0 else 0
    # 2. last partition cannot be extended downwards
    k_max_down = k_max_down if part_id < len(part)-1 else 0

    # Boundaries of partition
    i_begin, i_end = part[part_id]

    # Compute condition for upper neighbors (end shifted upwards)
    # This can be done in a seperate loop: only the condition of the target partition
    # needs to recomputed for all combinations of upwards and downwards shifts.
    if k_max_up > 0:
        i_upper_begin, i_upper_end = part[part_id-1]

        for k_up in range(0, k_max_up+1):
            mtx_upper_cond, mtx_upper = tridiag_cond_partition(
                mtx_fine, i_upper_begin, i_begin - k_up, n_halo)
            conds_shift_upper.append(mtx_upper_cond)
    
            print("Partition {} (A_PP [upshift {}, neigh], size {}x{}), condition {:e}".format(
                part_id-1, k_up, np.shape(mtx_upper)[0], np.shape(mtx_upper)[1], mtx_upper_cond))

    # Compute condition for lower neighbors (begin shifted downwards)
    if k_max_down > 0:
        i_lower_begin, i_lower_end = part[part_id+1]

        for k_down in range(0, k_max_down+1):
            mtx_lower_cond, mtx_lower = tridiag_cond_partition(
                mtx_fine, i_end + k_down, i_lower_end, n_halo)
            conds_shift_lower.append(mtx_lower_cond)
            
            print("Partition {} (A_PP [downshift {}, neigh], size {}x{}), condition {:e}".format(
                part_id+1, k_down, np.shape(mtx_lower)[0], np.shape(mtx_lower)[1], mtx_lower_cond))

    # Compute condition of partition (combination of up- and downshift)
    # Note: by including 0, the condition for the original bounds is included.
    # This way the original partition is included when computing the minimum
    # (in case shifting boundaries results in a higher condition).
    for k_up in range(0, k_max_up+1):
        for k_down in range(0, k_max_down+1):
            mtx_cond, mtx = tridiag_cond_partition(
                mtx_fine, i_begin - k_up, i_end + k_down, n_halo)
            conds_shift[(k_up, k_down)] = mtx_cond
            
            print("Partition {} (A_PP [upshift {}, downshift {}], size {}x{}), condition {:e}".format(
                part_id, k_up, k_down, np.shape(mtx)[0], np.shape(mtx)[1], mtx_cond))
            
            # Visualize partition with adjusted boundaries
            # plot_coarse_system(mtx, "partition, k_up = {}, k_down = {}, cond = {:e}".format(
            #     k_up, k_down, mtx_cond))
    
    # Compute minimal condition number for partition and its neighbors
    conds_shift_argmin = min(conds_shift, key=conds_shift.get)
    print("argmin k_up = {}, k_down = {}".format(
        conds_shift_argmin[0], conds_shift_argmin[1]))

    if k_max_up > 0:
        conds_shift_upper_argmin = np.argmin(conds_shift_upper)
        print("argmin k_up [neigh] = {}".format(conds_shift_upper_argmin))
     
    if k_max_down > 0:
        conds_shift_lower_argmin = np.argmin(conds_shift_lower)
        print("argmin k_down [neigh] = {}".format(conds_shift_lower_argmin))

    # Heuristic: if minimal neighbor has condition of a higher magnitude 
    # than the corresponding neighbor for the minimal partition, swap and check
    # XXX: this does not cover differences within the same order of magnitude
    # In some cases, an improvement in condition necessarily leads to a worse condition
    # of the neighbor (e.g. matrix 14), and we have to choose which to improve.
    conds_shift_argmin_k_up, conds_shift_argmin_k_down = conds_shift_argmin

    # Verify upper neighbor (k_down fixed)
    if k_max_up > 0:
        if conds_shift[(conds_shift_upper_argmin, conds_shift_argmin_k_down)] < 10*conds_shift[(conds_shift_argmin_k_up, conds_shift_argmin_k_down)]:
            if conds_shift_upper[conds_shift_argmin_k_up] < 10*conds_shift_upper[conds_shift_upper_argmin]:
                conds_shift_argmin_k_up = conds_shift_upper_argmin
        
        print("argmin k_up = {} [heuristic]".format(conds_shift_argmin_k_up))
    
    # Verify lower neighbor (k_up fixed)
    if k_max_down > 0:
        if (conds_shift[(conds_shift_argmin_k_up, conds_shift_lower_argmin)] < 10*conds_shift[(conds_shift_argmin_k_up, conds_shift_argmin_k_down)]):
            if conds_shift_lower[conds_shift_argmin_k_down] < 10*conds_shift_lower[conds_shift_lower_argmin]:
                conds_shift_argmin_k_down = conds_shift_lower_argmin
        
        print("argmin k_down = {} [heuristic]".format(conds_shift_argmin_k_down))

    conds_shift_argmin = (conds_shift_argmin_k_up, conds_shift_argmin_k_down)

    # Return new bounds
    new_cond = conds_shift[conds_shift_argmin]
    new_cond_upper = None
    new_cond_lower = None
    new_part[part_id] = [i_begin - conds_shift_argmin_k_up, i_end + conds_shift_argmin_k_down]

    if k_max_up > 0:
        new_cond_upper = conds_shift_upper[conds_shift_argmin_k_up]
        new_part[part_id-1] = [i_upper_begin, i_begin - conds_shift_argmin_k_up]

    if k_max_down > 0:
        new_cond_lower = conds_shift_lower[conds_shift_argmin_k_down]
        new_part[part_id+1] = [i_end + conds_shift_argmin_k_down, i_lower_end]

    #return new_part, conds_shift[conds_new_k], conds_shift_upper[conds_new_k]
    return new_part, new_cond, new_cond_upper, new_cond_lower


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