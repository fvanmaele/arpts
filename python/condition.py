#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 21:02:37 2022

@author: Ferdinand Vanmaele
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sys import stderr

import matplotlib # __version__
from packaging import version

TRIDIAG_VERBOSE=False


def generate_static_partition(N_fine, M):
    """
    Generate a partition of the index set [0, 1, ..., N_fine-1] where each
    partition has size M (if N divides M evenly), or the last partition has size N % M.

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

    # DivergingNorm in matplotlib 3.1, TwoSlopeNorm in 3.2    
    if version.parse(matplotlib.__version__) < version.parse("3.2"):
        norm_coarse = colors.DivergingNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    else:
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
    n_halo : int
        Include additional row and columns.

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

    return mtx_part_cond, mtx_part


# Note: invariants on part (half-open intervals)
# TODO: take ranges for k_max_up, k_max_down (shift upwards/downwards for both boundaries)
def tridiag_cond_shift(mtx_fine, part, part_id, n_halo, k_max_up=5, k_max_down=0):
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
    # XXX: the condition of the original partition (k_up = k_down = 0) is computed 3 times
    if k_max_up > 0:
        i_upper_begin, i_upper_end = part[part_id-1]

        for k_up in range(0, k_max_up+1):
            if i_begin - k_up <= i_upper_begin:
                conds_shift_upper.append(-np.Inf) # ensure index is ignored for heuristic below
                # print('warning: upper shift ignored (begin = {}, end = {})'.format(i_upper_begin, i_begin-k_up), file=stderr)
                continue

            mtx_upper_cond, mtx_upper = tridiag_cond_partition(
                mtx_fine, i_upper_begin, i_begin - k_up, n_halo)
            conds_shift_upper.append(mtx_upper_cond)
    
            if TRIDIAG_VERBOSE:
                print("Partition {} (A_PP [k_up = {}, upper], size {}x{}), condition {:e}".format(
                    part_id-1, k_up, np.shape(mtx_upper)[0], np.shape(mtx_upper)[1], mtx_upper_cond), file=stderr)

    # Compute condition for lower neighbors (begin shifted downwards)
    if k_max_down > 0:
        i_lower_begin, i_lower_end = part[part_id+1]
        
        for k_down in range(0, k_max_down+1):
            if i_end + k_down >= i_lower_end:
                conds_shift_lower.append(-np.Inf) # ensure index is ignored for heuristic below
                # print('warning: lower shift ignored (begin = {}, end = {})'.format(i_end+k_down, i_lower_end), file=stderr)
                continue

            mtx_lower_cond, mtx_lower = tridiag_cond_partition(
                mtx_fine, i_end + k_down, i_lower_end, n_halo)
            conds_shift_lower.append(mtx_lower_cond)
            
            if TRIDIAG_VERBOSE:
                print("Partition {} (A_PP [k_down = {}, lower], size {}x{}), condition {:e}".format(
                    part_id+1, k_down, np.shape(mtx_lower)[0], np.shape(mtx_lower)[1], mtx_lower_cond), file=stderr)

    # Compute condition of partition (combination of up- and downshift)
    # Note: by including 0, the condition for the original bounds is included.
    # This way the original partition is included when computing the minimum
    # (in case shifting boundaries results in a higher condition).
    for k_up in range(0, k_max_up+1):
        for k_down in range(0, k_max_down+1):
            mtx_cond, mtx = tridiag_cond_partition(
                mtx_fine, i_begin - k_up, i_end + k_down, n_halo)
            conds_shift[(k_up, k_down)] = mtx_cond

            if TRIDIAG_VERBOSE:
                print("Partition {} (A_PP [k_up = {}, k_down = {}], size {}x{}), condition {:e}".format(
                    part_id, k_up, k_down, np.shape(mtx)[0], np.shape(mtx)[1], mtx_cond), file=stderr)

            # Visualize partition with adjusted boundaries
            # plot_coarse_system(mtx, "partition, k_up = {}, k_down = {}, cond = {:e}".format(
            #     k_up, k_down, mtx_cond))
      
    # Compute minimal condition number for partition and its neighbors
    conds_shift_argmin = min(conds_shift, key=conds_shift.get)
    # print("argmin k_up = {}, k_down = {}".format(conds_shift_argmin[0], conds_shift_argmin[1]), file=stderr)
    if k_max_up > 0:
        conds_shift_upper_argmin = np.argmin(conds_shift_upper)
        # print("argmin k_up [neigh] = {}".format(conds_shift_upper_argmin), file=stderr)
    if k_max_down > 0:
        conds_shift_lower_argmin = np.argmin(conds_shift_lower)
        # print("argmin k_down [neigh] = {}".format(conds_shift_lower_argmin), file=stderr)

    # Heuristic: if minimal neighbor has condition of a higher magnitude 
    # than the corresponding neighbor for the minimal partition, swap and check
    # XXX: this does not cover differences within the same order of magnitude
    # In some cases, an improvement in condition necessarily leads to a worse condition
    # of the neighbor (e.g. matrix 14), and we have to choose which to improve.
    # TODO: take both factors (10, 10) as an argument
    conds_shift_argmin_k_up, conds_shift_argmin_k_down = conds_shift_argmin

    # Verify upper neighbor (k_down fixed)
    if k_max_up > 0:
        if conds_shift[(conds_shift_upper_argmin, conds_shift_argmin_k_down)] < 10*conds_shift[(conds_shift_argmin_k_up, conds_shift_argmin_k_down)]:
            if conds_shift_upper[conds_shift_argmin_k_up] < 10*conds_shift_upper[conds_shift_upper_argmin]:
                conds_shift_argmin_k_up = conds_shift_upper_argmin
        # print("argmin k_up = {} [heuristic]".format(conds_shift_argmin_k_up), file=stderr)
    
    # Verify lower neighbor (k_up fixed)
    if k_max_down > 0:
        if (conds_shift[(conds_shift_argmin_k_up, conds_shift_lower_argmin)] < 10*conds_shift[(conds_shift_argmin_k_up, conds_shift_argmin_k_down)]):
            if conds_shift_lower[conds_shift_argmin_k_down] < 10*conds_shift_lower[conds_shift_lower_argmin]:
                conds_shift_argmin_k_down = conds_shift_lower_argmin
        # print("argmin k_down = {} [heuristic]".format(conds_shift_argmin_k_down), file=stderr)

    conds_shift_argmin = (conds_shift_argmin_k_up, conds_shift_argmin_k_down)

    # Return new bounds
    cond_new = conds_shift[conds_shift_argmin]
    cond_new_upper = None
    cond_new_lower = None
    new_part[part_id] = [i_begin - conds_shift_argmin_k_up, i_end + conds_shift_argmin_k_down]

    if k_max_up > 0:
        cond_new_upper = conds_shift_upper[conds_shift_argmin_k_up]
        new_part[part_id-1] = [i_upper_begin, i_begin - conds_shift_argmin_k_up]

    if k_max_down > 0:
        cond_new_lower = conds_shift_lower[conds_shift_argmin_k_down]
        new_part[part_id+1] = [i_end + conds_shift_argmin_k_down, i_lower_end]

    return new_part, cond_new, cond_new_upper, cond_new_lower


def merge_partition(mtx_fine, partition, n_halo=1, min_length=1):
    new_partition = partition[:]
    part_len = len(partition)
    id_mask = [True]*part_len
    
    for part_id, idx in enumerate(partition):
        diff = idx[1] - idx[0]

        # TODO: set variable length
        if diff == 1:  # one element, a[i:i+1]
            # condition when merged with upper neighbor (if existing)
            if part_id == 0:
                # can only merge into lower partition
                i_lower_begin, i_lower_end = partition[part_id+1]
                new_partition[part_id+1] = i_lower_begin-1, i_lower_end

            elif part_id == part_len-1:
                # can only merge into upper partition
                i_upper_begin, i_upper_end = partition[part_id-1]
                new_partition[part_id-1] = i_upper_begin, i_upper_end+1     

            else:
                # decide on merge target depending on condition
                # XXX: should halo be passed here?
                i_lower_begin, i_lower_end = partition[part_id+1]
                i_upper_begin, i_upper_end = partition[part_id-1]
                
                cond_merge_with_lower = tridiag_cond_partition(
                        mtx_fine, i_lower_begin-1, i_lower_end, n_halo)
                cond_merge_with_upper = tridiag_cond_partition(
                        mtx_fine, i_upper_begin, i_upper_end+1, n_halo)
                
                if cond_merge_with_lower < cond_merge_with_upper:
                    new_partition[part_id+1] = i_lower_begin-1, i_lower_end
                else:
                    new_partition[part_id-1] = i_upper_begin, i_upper_end+1

            id_mask[part_id] = False

        elif diff == 0:
            id_mask[part_id] = False # no elements, a[i:i]
    
    # Remove elements that are masked from array
    new_partition = [new_partition[i] for i in range(0, len(id_mask)) if id_mask[i] is True]
    return new_partition, id_mask


# Note: Possible combinations of downshift and upshift for a single partition
# 1. compute upshift, compute downshift, take minimum
# 2. step-by-step increase (up+1, down+1, up+2, down+2, ...)
# 3. all possible combinations (up+1 x [down+{0,1,2,...}], up+2 x [down+{0,1,2,...}])
# 4. simple case: only compute upwards shifts

# Note: Possible orderings for adjusting boundaries
# a. Top to bottom, upwards shifts only: 
#    [0, 1, 2, 3, ...] -> upshift(0,*1); upshift(2,*3); ...
# b. Top to bottom, combine upwards and downwards shifts:
# c. Take partition with maximum condition (starting from static partitioning),
#    shift boundaries for it and upper neighbor
# d. As c., but also shift boundaries for lower neighbor
def tridiag_dynamic_partition(mtx_fine, static_partition, n_halo=1, k_max_up=5, k_max_down=0):
    n_partitions = len(static_partition)
    conds = []
    
    # Compute conditions for partitions in ascending order
    for part_id, idx in enumerate(static_partition):
        i_begin, i_end = idx
        mtx_part_cond, mtx_part = tridiag_cond_partition(
                mtx_fine, i_begin, i_end, n_halo)
        conds.append(mtx_part_cond)

        if TRIDIAG_VERBOSE:
            print("Partition {} (A_PP, size {}x{}), condition {:e}".format(
                part_id, np.shape(mtx_part)[0], np.shape(mtx_part)[1], mtx_part_cond), file=stderr)

    # Create mask for partition (True if boundaries were adjusted)
    partition_mask = [None] * n_partitions
    conds_decreasing = np.flip(np.argsort(conds)) # contains partition IDs with decreasing condition
    conds_argmax = conds_decreasing[0]
    # print("Maximum condition: Partition {} (A_PP) {:e}".format(conds_argmax, conds[conds_argmax]))

    # Keep track of conditions for shifted partitions for comparison purposes
    conds_adjusted = conds[:]
    dynamic_partition = static_partition[:]

    # Compute new bounds for partitions, starting with partition of maximum condition.
    # Includes upshifts and downshifts, so partitions are processed in triples.
    # (observation that there tend to be outliers of a very high condition)
    # XXX: Recompute maxima after adjusting partitions?
    for step in range(0, n_partitions):
        #conds_adjusted_decreasing = np.flip(np.argsort(conds_adjusted))
        conds_argmax_step = conds_decreasing[step] # partition number

        # Only process a partition and its neighbors once (*)
        # Adjust mask depending on k_max_up / k_max_down and partition number
        # XXX: still overlaps in some cases (bad thing or lucky accident?)
        mask_range = slice(0, 0)

        if conds_argmax_step > 0 and conds_argmax_step < n_partitions-1:
            if k_max_up > 0 and k_max_down > 0:
                mask_range = slice(conds_argmax_step-1, conds_argmax_step+2)
            elif k_max_up > 0:
                mask_range = slice(conds_argmax_step-1, conds_argmax_step+1)
            elif k_max_down > 0:
                mask_range = slice(conds_argmax_step, conds_argmax_step+2)               
        
        elif conds_argmax_step == n_partitions-1:
            if k_max_up > 0:
                mask_range = slice(conds_argmax_step-1, conds_argmax_step+1)

        elif conds_argmax_step == 0:
            if k_max_down > 0:
                mask_range = slice(conds_argmax_step, conds_argmax_step+2)

        # all([]) returns True, which occurs if mark_range = slice(0, 0)
        # XXX: make partition behavior variable
        if len(partition_mask[mask_range]) > 0 and all(partition_mask[mask_range]):
            continue
        # if partition_mask[part_id] is True:
        #     continue
        
        if TRIDIAG_VERBOSE:
            print("\nMaximum condition (step = {}): Partition {} (A_PP) {:e}".format(
                step, conds_argmax_step, conds[conds_argmax_step]), file=stderr)

        # Extend boundaries of partition upwards and downwards
        # TODO: assign result to temporary array, only assign if condition improved
        dynamic_partition, cond_new, cond_new_upper, cond_new_lower = tridiag_cond_shift(
            mtx_fine, dynamic_partition, conds_argmax_step, n_halo, k_max_up, k_max_down)
        if TRIDIAG_VERBOSE:
            print("Partition {} (A_PP, adjusted) {:e}".format(conds_argmax_step, cond_new), file=stderr)

        # Mark partition as processed
        partition_mask[conds_argmax_step] = True # partition

        # if conds[conds_argmax_step] < cond_new:
        #     print('Warning: repartitioning resulted in higher condition for partition {}'.format(
        #         conds_argmax_step), file=stderr)
        #     break
        conds_adjusted[conds_argmax_step] = cond_new

        # XXX: implicit verification of k_max_* > 0
        if cond_new_upper is not None:
            if TRIDIAG_VERBOSE:
                print("Partition {} (A_PP, adjusted) {:e}".format(conds_argmax_step-1, cond_new_upper), file=stderr)
            # Mask upper neighbor
            # partition_mask[conds_argmax_step-1] = True

            # if conds[conds_argmax_step-1] < cond_new_upper:
            #     print('Warning: repartitioning resulted in higher condition for upper partition {}'.format(
            #         conds_argmax_step-1), file=stderr)
            #     break
            conds_adjusted[conds_argmax_step-1] = cond_new_upper

        if cond_new_lower is not None:
            if TRIDIAG_VERBOSE:
                print("Partition {} (A_PP, adjusted) {:e}".format(conds_argmax_step+1, cond_new_lower), file=stderr)
            # Mask lower neighbor
            # partition_mask[conds_argmax_step+1] = True

            # if conds[conds_argmax_step+1] < cond_new_lower:
            #     print('Warning: repartitioning resulted in higher condition for lower partition {}'.format(
            #         conds_argmax_step+1), file=stderr) 
            #     break
            conds_adjusted[conds_argmax_step+1] = cond_new_lower
        
        if TRIDIAG_VERBOSE:
            print(partition_mask)

    conds_new_argmax = np.argmax(conds_adjusted)
    # print("Maximum condition (adjusted): Partition {} (A_PP) {:e}".format(
    #     conds_new_argmax, conds_adjusted[conds_new_argmax]))

    # Return additional values for parameter study
    return dynamic_partition, conds[conds_argmax], conds_adjusted[conds_new_argmax]


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