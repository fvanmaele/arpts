#!/usr/bin/env python
# This code is based on an implementation by Christoph Klein.
# Reference paper: "Tridiagonal GPU Solver with Scaled Partial Pivoting
# at Maximum Bandwidth, C. Klein, R. Strzodka, ICPP '21"
import numpy as np
from math import ceil
import sys
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
def rptapp(a_fine, b_fine, c_fine, d_fine, M, N_tilde, 
           k_max_up=0, k_max_down=0, threshold=0, n_halo=1, level=0):
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
    static_partition = condition.generate_static_partition(N_fine, M)
    # XXX: show warnings from this step at program end
    dynamic_partition = condition.tridiag_dynamic_partition(
        mtx_fine, static_partition, n_halo, k_max_up, k_max_down)

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
    print("\nCoarse system (M = {}, k_max_up = {}, k_max_down = {}), condition: {:e}".format(
        M, k_max_up, k_max_down, mtx_coarse_cond))

    # Plot coarse system
    # TODO: add condition number to title
    condition.plot_coarse_system(mtx_coarse)

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
        x_coarse = rptapp(a_coarse, b_coarse, c_coarse, d_coarse, M, N_tilde, 
                          k_max_up, k_max_down, threshold, n_halo, level=level+1)    

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
    k_max_up = 0
    k_max_down = 1
    n_halo = 1
    x_fine_rptapp = rptapp(a_fine, b_fine, c_fine, d_fine, M, N_tilde, k_max_up, k_max_down, 0, n_halo)

    print("FRE = ", np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine))
    print("\n")
    #print("xfine_calculated:\n", x_fine_rptapp)
