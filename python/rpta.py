#!/usr/bin/env python
# This code is based on an implementation by Christoph Klein.
# Reference paper: "Tridiagonal GPU Solver with Scaled Partial Pivoting
# at Maximum Bandwidth, C. Klein, R. Strzodka, ICPP '21"
import numpy as np
from math import ceil
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import condition

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


def rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse,
                  d_coarse, M, threshold=0):
    N = len(a_fine)
    partition_start = 0

    for partition_id, partition_offset in enumerate(range(0, N, M)):
        partition_end = min(partition_offset +M, N)
        #partition_size = partition_end - partition_start;
        
        a_coarse_lower, b_coarse_lower, c_coarse_lower, d_coarse_lower = eliminate_band(
            a_fine[partition_offset:partition_end],
            b_fine[partition_offset:partition_end],
            c_fine[partition_offset:partition_end],
            d_fine[partition_offset:partition_end],
            threshold)
        c_coarse_upper, b_coarse_upper, a_coarse_upper, d_coarse_upper = eliminate_band(
            list(reversed(c_fine[partition_offset:partition_end])),
            list(reversed(b_fine[partition_offset:partition_end])),
            list(reversed(a_fine[partition_offset:partition_end])),
            list(reversed(d_fine[partition_offset:partition_end])),
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


def plot_coarse_system(mtx_coarse, title='Coarse system'):
    vmin=np.min(mtx_coarse)
    vmax=np.max(mtx_coarse)
    vcenter = 0 if vmin < 0 else (vmin+vmax)/2
    norm_coarse = colors.TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)

    plt.matshow(mtx_coarse, cmap='bwr', norm=norm_coarse)
    plt.title(title)
    plt.colorbar()
    plt.show()


def rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, M, threshold=0):
    N = len(a_fine)
    num_partitions = N / M
    x_fine = [None] * N

    for partition_id, partition_offset in enumerate(range(0, N, M)):
        partition_end = min(partition_offset+M, N)

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
            list(a_fine[partition_offset:partition_end]),
            list(b_fine[partition_offset:partition_end]),
            list(c_fine[partition_offset:partition_end]),
            list(d_fine[partition_offset:partition_end]),
            x1_prev_partition, x0, x1, x0_next_partition, threshold)

        x_fine[partition_offset:partition_end] = x_partition

    return x_fine


# TODO: cover the case where M does not divide N, and we have a
# partition of size N mod M (more options to set partition boundaries,
# for adaptive partitioning)
# "it must be true that `num_partitions_per_block` <= block dimension"
def rptapp(a_fine, b_fine, c_fine, d_fine, M, N_tilde, threshold=0, halo_n=1, level=0):
    N_fine = len(a_fine)
    N_coarse = (ceil(N_fine / M)) * 2
    #assert(N_fine == len(b_fine) == len(c_fine) == len(d_fine))
    #assert(N_coarse * M == N_fine * 2)

    print("recursion level: {}".format(level))
    print("dim. fine system: {}".format(N_fine))
    print("dim. coarse system: {}".format(N_coarse))
    print("partition size: {}".format(M))
    
    # Compute condition of fine system (only for numerical experiments)
    mtx_fine = bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    mtx_cond = np.linalg.cond(mtx_fine)
    print("Fine system, condition: {:e}".format(mtx_cond))

    # -------------------------------------------------------------------------
    # Compute (maximum) condition of partitions
    conds = []
    if N_fine // M == int(N_fine / M):
        n_partitions = (N_fine // M)
    else:
        n_partitions = (N_fine // M) + 1

    # Compute conditions for partitions of size M
    # TODO: make this work with dynamic bounds, not just static ones
    #       take an array of begin/start indices per partition?
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

        # i_begin = i*M
        # i_end = (i+1)*M

        mtx_part = mtx_fine[i_begin:i_end, i_begin:i_end]
        mtx_part_cond = np.linalg.cond(mtx_part)
        mtx_part_shape = np.shape(mtx_part)
        conds.append(mtx_part_cond)

        print("Partition {} (A_PP, size {}x{}), condition {:e}".format(
            i, mtx_part_shape[0], mtx_part_shape[1], mtx_part_cond))

    # Compute condition for remaining partition of size N % M
    if N_fine % M > 0:
        i = n_partitions
        # subtract 1 for nodes A_IP (upper halo)
        i_begin = i*M - halo_n
        # i_begin = i*M
        i_end = N_fine

        mtx_part = mtx_fine[i_begin:i_end, i_begin:i_end]
        mtx_part_cond = np.linalg.cond(mtx_part)
        mtx_part_shape = np.shape(mtx_part)
        conds.append(mtx_part_cond)

        print("Partition {} (A_PP, size {}x{}), condition {:e}".format(
            i, mtx_part_shape[0], mtx_part_shape[1], mtx_part_cond))

    # Compute partition index of maximum condition
    part_max_cond_i = np.argmax(conds)
    print("Maximum condition: Partition {} (A_PP) {:e}".format(
        part_max_cond_i, conds[part_max_cond_i]))

    # Extend boundaries of partition upwards
    # TODO: also extend boundaries of upper neighbor
    # TODO: mark partition (and neighbors?) as "done" when minimum condition is found, proceed to next
    # TODO: include halo in condition calculations?
    # TODO: move this to separate function
    # TODO: Generalize for both upshift and downshift
    if part_max_cond_i > 0:
        conds_upshift = []
        conds_upshift_neigh = []

        for k_up in range(0, 6):
            # boundaries of block
            i_new_begin = part_max_cond_i*M - k_up
            i_new_end = (part_max_cond_i+1)*M

            mtx_new = mtx_fine[i_new_begin:i_new_end, i_new_begin:i_new_end]
            mtx_new_cond = np.linalg.cond(mtx_new)
            mtx_new_shape = np.shape(mtx_new)
            conds_upshift.append(mtx_new_cond)

            print("Partition {} (A_PP [upshift {}], size {}x{}), condition {:e}".format(
                part_max_cond_i, k_up, mtx_new_shape[0], mtx_new_shape[1], mtx_new_cond))

            # boundaries of upper neighbor
            i_neigh_begin = max(0, (part_max_cond_i-1)*M)
            i_neigh_end = i_new_begin - 1

            mtx_neigh = mtx_fine[i_neigh_begin:i_neigh_end, i_neigh_begin:i_neigh_end]
            mtx_neigh_cond = np.linalg.cond(mtx_neigh)
            mtx_neigh_shape = np.shape(mtx_neigh)
            conds_upshift_neigh.append(mtx_neigh_cond)
            
            print("Partition {} (A_PP [upshift {}, neigh], size {}x{}), condition {:e}".format(
                part_max_cond_i-1, k_up, mtx_neigh_shape[0], mtx_neigh_shape[1], mtx_neigh_cond))
    
            # Plot partitions
            plot_coarse_system(mtx_new, "partition, k_up = {}, cond = {:e}".format(
                k_up, mtx_new_cond))
            # plot_coarse_system(mtx_neigh, "partition, k_up = {} [neigh], cond = {:e}".format(
            #     k_up, mtx_neigh_cond))

        conds_upshift_min_k = np.argmin(conds_upshift)
        print("argmin k_up = {}".format(conds_upshift_min_k))
        
        conds_upshift_neigh_min_k = np.argmin(conds_upshift_neigh)
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


    # Extend boundaries of partition downwards
    # TODO: also extend boundaries of lower neighbor
    # TODO: mark partition (and neighbors?) as "done" when minimum condition is found
    # TODO: include halo in condition calculations?
    # TODO: move this to separate function
    if part_max_cond_i < n_partitions:
        conds_downshift = []
        conds_downshift_neigh = []
        
        for k_down in range(0, 6):
            # boundaries of block
            i_new_begin = part_max_cond_i*M
            i_new_end = (part_max_cond_i+1)*M + k_down

            mtx_new = mtx_fine[i_new_begin:i_new_end, i_new_begin:i_new_end]
            mtx_new_cond = np.linalg.cond(mtx_new)
            conds_downshift.append(mtx_new_cond)
            mtx_new_shape = np.shape(mtx_new)
    
            print("Partition {} (A_PP [downshift {}], size {}x{}), condition {:e}".format(
                part_max_cond_i, k_down, mtx_new_shape[0], mtx_new_shape[1], mtx_new_cond))
    
            # boundaries of lower neighbor
            i_neigh_begin = i_new_end + 1
            i_neigh_end = min(N_fine, (part_max_cond_i+2)*M)

            # Plot partitions
            plot_coarse_system(mtx_new, "partition, k_down = {}, cond = {:e}".format(
                k_down, mtx_new_cond))
        
        conds_downshift_min_k = np.argmin(conds_downshift)
        print("argmin k_down = {}".format(conds_downshift_min_k))

    # -------------------------------------------------------------------------
    # Reduce to coarse system
    a_coarse = np.zeros(N_coarse)
    b_coarse = np.zeros(N_coarse)
    c_coarse = np.zeros(N_coarse)
    d_coarse = np.zeros(N_coarse)

    rptapp_reduce(a_fine, b_fine, c_fine, d_fine,
                  a_coarse, b_coarse, c_coarse, d_coarse, M, threshold)
    mtx_coarse = bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)

    # Compute condition of coarse system
    mtx_coarse_cond = np.linalg.cond(mtx_coarse)
    print("Coarse system (M = {}), condition: {:e}".format(M, mtx_coarse_cond))

    # Plot coarse system
    plot_coarse_system(mtx_coarse)

    # If system size below threshold, solve directly
    if len(a_coarse) <= N_tilde:
        x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
    # Otherwise, do recursive call
    else:
        x_coarse = rptapp(a_coarse, b_coarse, c_coarse, d_coarse,
                          M, N_tilde, threshold, halo_n, level=level+1)    

    # Substitute into fine system
    x_fine_rptapp = rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, 
                                      M, threshold)
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
Ms = [33]

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
    halo_n = 0
    threshold = 0
    x_fine_rptapp = rptapp(a_fine, b_fine, c_fine, d_fine, M, N_tilde, threshold, halo_n)

    print("FRE = ", np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine))
    print("\n")
    #print("xfine_calculated:\n", x_fine_rptapp)
