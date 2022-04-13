#!/usr/bin/env python
# This code is based on an implementation by Christoph Klein.
# Reference paper: "Tridiagonal GPU Solver with Scaled Partial Pivoting
# at Maximum Bandwidth, C. Klein, R. Strzodka, ICPP '21"
import numpy as np
from math import ceil
import sys
import warnings

from condition import generate_static_partition, tridiag_cond_shift
from condition import plot_coarse_system, tridiag_cond_partition

def apply_threshold(x, y, eps):
    xt, yt = [0, 0]
    if eps == 0:
        return xt, yt
    realmin = sys.float_info.epsilon
    if abs(x) < eps:
        xt = realmin
    if abs(y) < eps:
        yt = 0
    return xt, yt


def read_matrixmarket(filename):
    with open(filename) as f:
        s = f.read()
        ss = s.split('\n')
        l = ss[1].split()
        num_rows = int(l[0])
        num_cols = int(l[1])
        arr = np.zeros((num_rows, num_cols))
        
        for l in ss[2:-1]:
            i, j, x = list(map(float, l.split()))
            arr[int(i) - 1][int(j) - 1] = x

        return arr
    

def read_dense_matrixmarket(filename):
    with open(filename) as f:
        s = f.read()
        ss = s.split('\n')
        l = ss[1].split()
        num_rows = int(l[0])
        # TODO can only read a vector
        num_cols = int(l[1])
        arr = np.zeros(num_rows)
        
        for i,l in enumerate(ss[2:-1]):
            x = float(l)
            arr[i] = x

        return arr


def bands_to_numpy_matrix(a, b, c):
    N = len(a)
    mtx = np.zeros((N, N))
    
    for row_id in range(N):
        mtx[row_id][row_id] = b[row_id]
        if row_id > 0:
            mtx[row_id][row_id - 1] = a[row_id]
        if row_id < N - 1:
            mtx[row_id][row_id + 1] = c[row_id]

    return mtx


def numpy_matrix_to_bands(mtx):
    N = mtx.shape[0]
    a = [0.0] * N
    b = [0.0] * N
    c = [0.0] * N
    
    for row_id in range(N):
        b[row_id] = mtx[row_id][row_id]
        if row_id > 0:
            a[row_id] = mtx[row_id][row_id - 1]
        if row_id < N - 1:
            c[row_id] = mtx[row_id][row_id + 1]

    return a, b, c

# TODO: add visualization for elements below spike
def eliminate_band(a, b, c, d, threshold=0):
    #print("eliminate_band: threshold: {}".format(threshold))
    M = len(a)
    # to save a, b, c, d, spike
    s_p = [0.0] * 5
    s_c = [0.0] * 5

    s_p[0] = a[1]
    s_p[1] = b[1]
    s_p[2] = c[1]
    s_p[3] = 0.0
    s_p[4] = d[1]

    for j in range(2, M):
        s_c[0] = 0.0
        s_c[1] = a[j]
        s_c[2] = b[j]
        s_c[3] = c[j]
        s_c[4] = d[j]

        apply_threshold(s_p[1], s_c[1], threshold)

        if abs(s_c[1]) > abs(s_p[1]):
            r_c = (-1.0) * s_p[1] / s_c[1]
            r_p = 1.0
        else:
            r_c = 1.0
            r_p = (-1.0) * s_c[1] / s_p[1]

        for k in [0, 2, 3, 4]:
            s_p[k] = r_p * s_p[k] + r_c * s_c[k]

        s_p[1] = s_p[2]
        s_p[2] = s_p[3]
        s_p[3] = 0.0

    return s_p[0], s_p[1], s_p[2], s_p[4]


# TODO: adjust for partitions with dynamic boundaries (take list as input)
def rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse,
                  d_coarse, partition, threshold=0):
    # num_partitions = len(partition)
    # N = len(a_fine)
    # partition_start = 0

    for partition_id, partition_bounds in enumerate(partition):
    # for partition_id, partition_offset in enumerate(range(0, N, M)):
        partition_begin = partition_bounds[0]
        partition_end = partition_bounds[1]
        # partition_end = min(partition_offset +M, N)
        # partition_size = partition_end - partition_start;
        
        a_coarse_lower, b_coarse_lower, c_coarse_lower, d_coarse_lower = eliminate_band(
            a_fine[partition_begin:partition_end],
            b_fine[partition_begin:partition_end],
            c_fine[partition_begin:partition_end],
            d_fine[partition_begin:partition_end],
            threshold)
        c_coarse_upper, b_coarse_upper, a_coarse_upper, d_coarse_upper = eliminate_band(
            list(reversed(c_fine[partition_begin:partition_end])),
            list(reversed(b_fine[partition_begin:partition_end])),
            list(reversed(a_fine[partition_begin:partition_end])),
            list(reversed(d_fine[partition_begin:partition_end])),
            threshold)

        a_coarse[2 * partition_id] = a_coarse_upper
        b_coarse[2 * partition_id] = b_coarse_upper
        c_coarse[2 * partition_id] = c_coarse_upper
        d_coarse[2 * partition_id] = d_coarse_upper

        a_coarse[2 * partition_id + 1] = a_coarse_lower
        b_coarse[2 * partition_id + 1] = b_coarse_lower
        c_coarse[2 * partition_id + 1] = c_coarse_lower
        d_coarse[2 * partition_id + 1] = d_coarse_lower


# x1_prev_partition = 0 for the first partition
# x0_next_partition = 0 for the last partition
def eliminate_band_with_solution(a, b, c, d, x1_prev_partition, x0, x1,
                                 x0_next_partition, threshold=0):
    #print("eliminate_band_with_solution: threshold: {}".format(threshold))
    M = len(a)

    x = np.zeros(M)
    x[M - 1] = x1
    x[0] = x0

    # Substitute with solution
    d[M - 2] = d[M - 2] - c[M - 2] * x1
    c[M - 2] = 0.0

    s_c = [0.0] * 5
    s_p = [0.0] * 5

    s_p[0] = 0.0
    s_p[1] = b[1]
    s_p[2] = c[1]
    s_p[3] = 0.0
    s_p[4] = d[1] - a[1] * x0
    ip = 1
    i = [None] * M

    # DOWNWARDS ORIENTED ELIMINATION
    for j in range(2, M - 1):
        s_c[0] = 0.0
        s_c[1] = a[j]
        s_c[2] = b[j]
        s_c[3] = c[j]
        s_c[4] = d[j]
        apply_threshold(s_p[1], s_c[1], threshold)

        if abs(s_c[1]) > abs(s_p[1]):
            i[j - 1] = j
            r_c = (-1.0) * s_p[1] / s_c[1]
            r_p = 1.0
        else:
            i[j - 1] = ip
            r_c = 1.0
            r_p = (-1.0) * s_c[1] / s_p[1]
            a[ip] = s_p[1]
            b[ip] = s_p[2]
            c[ip] = 0.0
            d[ip] = s_p[4]
            ip = j

        for k in [2, 3, 4]:
            s_p[k] = r_p * s_p[k] + r_c * s_c[k]

        s_p[1] = s_p[2]
        s_p[2] = s_p[3]
        s_p[3] = 0.0

    if abs(s_p[1]) < abs(a[M - 1]):
        x[M - 2] = (d[M - 1] - b[M - 1] * x[M - 1] -
                    c[M - 1] * x0_next_partition) / a[M - 1]
    else:
        x[M - 2] = s_p[4] / s_p[1]

    # UPWARDS ORIENTED SUBSTITUTION
    for j in reversed(range(1, M - 2)):
        k = i[j]
        x[j] = (d[k] - b[k] * x[j + 1] - c[k] * x[j + 2]) / a[k]

    k = i[1]
    # print(i)
    # print(a)
    # print(x)
    # print("--")

    # FIXME: "None" k in last partition of size N % M
    if k is not None and abs(a[k]) < abs(c[0]):
        x[1] = (d[0] - a[0] * x1_prev_partition - b[0] * x[0]) / c[0]
    elif k is not None:
        x[1] = (d[k] - b[k] * x[2] - c[k] * x[3]) / a[k]

    return x


def rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, partition, threshold=0):
    num_partitions = len(partition)
    N = len(a_fine)
    x_fine = [None] * N

    for partition_id, partition_bounds in enumerate(partition):
    # for partition_id, partition_offset in enumerate(range(0, N, M)):
        partition_begin = partition_bounds[0]
        partition_end = partition_bounds[1]
        # partition_end = min(partition_offset+M, N)

        if partition_id > 0:
            x1_prev_partition = x_coarse[partition_id * 2 - 1]
        else:
            x1_prev_partition = 0.0

        if partition_id < num_partitions - 1:
            x0_next_partition = x_coarse[(partition_id + 1) * 2]
        else:
            x0_next_partition = 0.0

        x0 = x_coarse[partition_id * 2]
        x1 = x_coarse[partition_id * 2 + 1]

        x_partition = eliminate_band_with_solution(
            list(a_fine[partition_begin:partition_end]),
            list(b_fine[partition_begin:partition_end]),
            list(c_fine[partition_begin:partition_end]),
            list(d_fine[partition_begin:partition_end]),
            x1_prev_partition, x0, x1, x0_next_partition, threshold)

        x_fine[partition_begin:partition_end] = x_partition

    return x_fine


# TODO: cover the case where M does not divide N, and we have a
# partition of size N mod M (more options to set partition boundaries,
# for adaptive partitioning)
# "it must be true that `num_partitions_per_block` <= block dimension"
def rptapp(a_fine, b_fine, c_fine, d_fine, M, N_tilde, threshold=0, n_halo=1, level=0):
    N_fine = len(a_fine)
    N_coarse = (ceil(N_fine / M)) * 2
    #assert(N_fine == len(b_fine) == len(c_fine) == len(d_fine))
    #assert(N_coarse * M == N_fine * 2)

    print("recursion level: {}".format(level))
    print("dim. fine system: {}".format(N_fine))
    print("dim. coarse system: {}".format(N_coarse))
    print("partition size: {}".format(M))
    print("halo size: {}".format(n_halo))

    # Compute condition of fine system (only for numerical experiments)
    mtx_fine = bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    mtx_cond = np.linalg.cond(mtx_fine)
    print("Fine system, condition: {:e}".format(mtx_cond))

    # -------------------------------------------------------------------------
    # Compute (maximum) condition for partitions of a fixed size
    # TODO: move generation of dynamic partition to a function

    conds = []
    static_partition = generate_static_partition(N_fine, M)
    n_partitions = len(static_partition)
    
    # Take halo into account when computing condition
    # if i > 0:
    #     # subtract 1 for nodes A_IP (upper halo), i_begin == 0
    #     i_begin = i*M - halo_n
    # else:
    #     i_begin = i*M
    
    # if i == n_partitions-1:
    #     # we are in the last partition, of size M, i_end == N_fine
    #     i_end = (i+1)*M
    # else:
    #     # we are in an inner partition, of size M
    #     i_end = (i+1)*M + halo_n

    # Compute conditions for partitions in ascending order
    for part_i, idx in enumerate(static_partition):
        i_begin, i_end = idx
        mtx_part_cond, mtx_part = tridiag_cond_partition(mtx_fine, i_begin, i_end, n_halo)
        conds.append(mtx_part_cond)

        print("Partition {} (A_PP, size {}x{}), condition {:e}".format(
            part_i, np.shape(mtx_part)[0], np.shape(mtx_part)[1], mtx_part_cond))

    # Create mask for partition (True if boundaries were adjusted)
    # XXX: When shifting partition boundaries, the number of partitions remains fixed.
    # Allow to "merge" one partition into the other, reducing the number of partitions?
    partition_mask = [None] * n_partitions
    conds_decreasing = np.flip(np.argsort(conds)) # contains partition IDs with decreasing condition
    conds_argmax = conds_decreasing[0]
    
    print("Maximum condition: Partition {} (A_PP) {:e}".format(
        conds_argmax, conds[conds_argmax]))

    print(static_partition)
    dynamic_partition = static_partition[:]

    # TODO: maximum amount of row/column shifts, make this a parameter
    k_max_up = 5
    k_max_down = 0

    # XXX: Possible combinations of downshift and upshift for a single partition
    # 1. compute upshift, compute downshift, take minimum
    # 2. step-by-step increase (up+1, down+1, up+2, down+2, ...)
    # 3. all possible combinations (up+1 x [down+{0,1,2,...}], up+2 x [down+{0,1,2,...}])
    # 4. simple case: only compute upwards shifts
    
    # XXX: Possible orderings for adjusting boundaries
    # a. Top to bottom, upwards shifts only: 
    #    [0, 1, 2, 3, ...] -> upshift(0,*1); upshift(2,*3); ...
    # b. Top to bottom, combine upwards and downwards shifts:
    # c. Take partition with maximum condition (starting from static partitioning),
    #    shift boundaries for it and upper neighbor
    # d. As c., but also shift boundaries for lower neighbor

    # Compute new bounds for partitions, starting with partition of maximum condition.
    # Includes upshifts and downshifts, so partitions are processed in triples.
    # (observation that there tend to be outliers of a very high condition)
    # XXX: Recompute maxima after adjusting partitions?
    for step in range(0, n_partitions-1):
        conds_argmax_step = conds_decreasing[step] # partition number

        # Only process a partition and its neighbors once (*)
        # Adjust mask depending on k_max_up / k_max_down and partition number
        if step > 0 and step < n_partitions-1:
            if k_max_up > 0 and k_max_down > 0:
                mask_range = slice(conds_argmax_step-1, conds_argmax_step+2)
            elif k_max_up > 0:
                mask_range = slice(conds_argmax_step-1, conds_argmax_step+1)
            elif k_max_down > 0:
                mask_range = slice(conds_argmax_step, conds_argmax_step+2)               
        elif step == n_partitions-1:
            if k_max_up > 0:
                mask_range = slice(conds_argmax_step-1, conds_argmax_step+1)
        else:
            mask_range = slice(0, 0)

        if any(partition_mask[mask_range]):
            continue

        print("Maximum condition (step = {}): Partition {} (A_PP) {:e}".format(
            step, conds_argmax_step, conds[conds_argmax_step]))

        # Extend boundaries of partition upwards and downwards
        # Note: different elements of dynamic_partition are processed in each step (by (*))
        dynamic_partition, new_cond, new_cond_upper, new_cond_lower = tridiag_cond_shift(
            mtx_fine, dynamic_partition, conds_argmax_step, n_halo, k_max_up, k_max_down)

        # Mark partition and neighbors as processed
        partition_mask[conds_argmax_step] = True # partition    
        print("Partition {} (A_PP, adjusted) {:e}".format(conds_argmax_step, new_cond))

        # XXX: instead of aborting the algorithm when the (neighboring) partition
        # has a higher condition, mark it as False and process it again 
        # (in the direction towards partitions which were not improved?)
        if conds[conds_argmax_step] < new_cond:
            warnings.warn('repartitioning resulted in higher condition for partition {}'.format(
                conds_argmax_step), RuntimeWarning)
            break

        # XXX: implicit verification of k_max_* > 0
        if new_cond_upper is not None:
            partition_mask[conds_argmax_step-1] = True # upper neighbor
            print("Partition {} (A_PP, adjusted) {:e}".format(conds_argmax_step-1, new_cond_upper))
            
            if conds[conds_argmax_step] < new_cond_upper:
                warnings.warn('repartitioning resulted in higher condition for partition {}'.format(
                    conds_argmax_step-1), RuntimeWarning)
                break

        if new_cond_lower is not None:
            partition_mask[conds_argmax_step+1] = True # lower neighbor
            print("Partition {} (A_PP, adjusted) {:e}".format(conds_argmax_step+1, new_cond_lower))

            if conds[conds_argmax_step] < new_cond_lower:
                warnings.warn('repartitioning resulted in higher condition for partition {}'.format(
                    conds_argmax_step+1), RuntimeWarning)      
                break
        
        print(partition_mask)
        print(dynamic_partition)

    # -------------------------------------------------------------------------
    # Reduce to coarse system
    a_coarse = np.zeros(N_coarse)
    b_coarse = np.zeros(N_coarse)
    c_coarse = np.zeros(N_coarse)
    d_coarse = np.zeros(N_coarse)

    # XXX: Do this in each repartitioning step, and plot the values
    rptapp_partition = dynamic_partition # Change partition here for comparison purposes
    
    # TODO: document input/output parameters of rptapp_reduce()
    rptapp_reduce(a_fine, b_fine, c_fine, d_fine,
                  a_coarse, b_coarse, c_coarse, d_coarse, 
                  rptapp_partition, threshold)
    mtx_coarse = bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)

    # Compute condition of coarse system
    mtx_coarse_cond = np.linalg.cond(mtx_coarse)
    print("Coarse system (M = {}), condition: {:e}".format(M, mtx_coarse_cond))

    # Plot coarse system
    plot_coarse_system(mtx_coarse)

    # If system size below threshold, solve directly
    # TODO: the case <= N_tilde always hold, add parameter
    if len(a_coarse) <= N_tilde:
        x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
    # Otherwise, do recursive call
    # Note: each recursive call starts from a static partition, and computes
    # new boundaries based on condition of the partitions
    # TODO: set maximum level for adjusting partition boundaries?
    # TODO: pass newly computed M, N_tilde to recursive call (coarse system) 
    else:
        sys.exit('function not implemented')
        x_coarse = rptapp(a_coarse, b_coarse, c_coarse, d_coarse,
                          M, N_tilde, threshold, n_halo, level=level+1)    

    # Substitute into fine system
    x_fine_rptapp = rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, 
                                      rptapp_partition, threshold)
    return x_fine_rptapp

        
# # lower diagonal
# a_fine = np.random.rand(N_fine)
# # main diagonal
# b_fine = np.random.rand(N_fine)
# # upper diagonal
# c_fine = np.random.rand(N_fine)

#12.mtx  14.mtx  15.mtx  16.mtx  17.mtx  18.mtx  19.mtx  1.mtx  20.mtx  2.mtx  4.mtx  6.mtx 
mtx_number = 14
mtx_dim = 512
fn = "../mtx/{}-{}.mtx".format(mtx_number, mtx_dim)
mtx = read_matrixmarket(fn)
a_fine, b_fine, c_fine = numpy_matrix_to_bands(mtx)

# TODO: add command-line arguments (matrix ID/file/dimension, block size M,
# recursion limit N_tilda, threshold)
N_fine = mtx.shape[0]
#Ms = [4, 8, 16, 32, 64]
#Ms = range(16, 33)
Ms = [32]

for M in Ms:
    # N_coarse = (N_fine // M) * 2
    # assert(N_coarse * M == N_fine * 2)

    # print("dim. fine system: {}".format(N_fine))
    # print("dim. coarse system: {}".format(N_coarse))
    # print("partition size: {}".format(M))

    # solution
    np.random.seed(0)
    x_fine = np.random.normal(3, 1, N_fine)
    #x_fine = read_dense_matrixmarket("../numerical-test-matrices/N-14-x.mtx")

    #mtx = bands_to_numpy_matrix(a_fine, b_fine, c_fine)

    # rhs
    d_fine = np.matmul(mtx, x_fine)

    #N_tilde = 64
    N_tilde = (ceil(N_fine / M)) * 2
    #print("N_tilde: {}".format(N_tilde))
    n_halo = 1
    threshold = 0
    x_fine_rptapp = rptapp(a_fine, b_fine, c_fine, d_fine, M, N_tilde, threshold, n_halo)

    print("FRE = ", np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine))
    print("\n")
    #print("xfine_calculated:\n", x_fine_rptapp)
