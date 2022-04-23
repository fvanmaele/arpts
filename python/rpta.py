#!/usr/bin/env python
# This code is based on an implementation by Christoph Klein.
# Reference paper: "Tridiagonal GPU Solver with Scaled Partial Pivoting
# at Maximum Bandwidth, C. Klein, R. Strzodka, ICPP '21"

# %% Libraries
import numpy as np
from math import ceil
import sys

import condition
import matrix

# %% Functions
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


# TODO: add visualization for elements below spike
def eliminate_band(a, b, c, d, threshold=0):
    M = len(a)
    assert(M > 1) # band should at least have one element

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
    assert(M > 1) # band should at least have one element

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


def rptapp_generate_partition(a_fine, b_fine, c_fine, M, n_halo, k_max_up, k_max_down):
    N_fine = len(a_fine)
    N_coarse = (ceil(N_fine / M)) * 2

    # Compute condition of fine system (only for numerical experiments)
    mtx_fine = matrix.bands_to_numpy_matrix(a_fine, b_fine, c_fine)
    mtx_cond = np.linalg.cond(mtx_fine)
    # print("Fine system, condition: {:e}".format(mtx_cond))

    # Start with partition with blocks of fixed size
    static_partition = condition.generate_static_partition(N_fine, M)

    dyn_partition, mtx_cond_partmax, mtx_cond_partmax_dyn = condition.tridiag_dynamic_partition(
        mtx_fine, static_partition, n_halo, k_max_up, k_max_down)
    assert(N_coarse == len(dyn_partition)*2)
        
    # Merge partitions with only a single element
    # If multiple neighbors are available, choose based on minimal condition
    dyn_partition, id_mask = condition.merge_partition(mtx_fine, dyn_partition, n_halo)
    if not all(id_mask):
        print('warning: changed partition size from {} to {}'.format(
            N_coarse / 2, len(dyn_partition)), file=sys.stderr)
    N_coarse = len(dyn_partition)*2
    # If system size below threshold, solve directly
    # TODO: the case <= N_tilde always hold, add parameter
    return dyn_partition, mtx_cond, mtx_cond_partmax, mtx_cond_partmax_dyn


def rptapp_generate_partition_det(a_fine, b_fine, c_fine, part_min, part_max):
    N = len(a_fine)
    assert(part_min < part_max)
    assert(part_min > 0)
    assert(part_max < N-1)
    
    partition = []
    i_begin = 0 # Index of first row (upper boundary)    
    while i_begin + part_max < N:
        dets = []
        for offset in range(part_min, part_max):
            i_target = min(i_begin + offset, N-1)
            mtx = np.matrix([[b_fine[i_begin], c_fine[i_begin]], 
                             [a_fine[i_target], b_fine[i_target]]])
            dets.append(abs(np.linalg.det(mtx)))
            # print("{}, {}: |det| = {}".format(
            #     i_begin, i_target, abs(np.linalg.det(mtx))))
        
        # Criterion: maximum determinant
        dets_argmax = np.argmax(dets)
        print("{}, {}: |det| (max) = {}".format(
            i_begin, i_begin + part_min + dets_argmax, dets[dets_argmax]))
        partition.append([i_begin, i_begin + part_min + dets_argmax])

        # Go to next partition
        i_begin = min(i_begin + part_min + dets_argmax, N)
    
    # Append last partition
    if i_begin < N:
        mtx = np.matrix([[b_fine[i_begin], c_fine[i_begin]], 
                         [a_fine[N-1], b_fine[N-1]]])
        det = abs(np.linalg.det(mtx))
        print("{}, {}: |det| = {}".format(i_begin, N-1, det))
        partition.append([i_begin, N])

    return partition


# XXX: "it must be true that `num_partitions_per_block` <= block dimension"
# TODO: move condition logic to a separate function if possible
def rptapp(a_fine, b_fine, c_fine, d_fine, N_tilde, M,
           k_max_up=0, k_max_down=0, threshold=0, n_halo=1, level=0):
    # N_fine = len(a_fine)
    # N_coarse = (ceil(N_fine / M)) * 2

    # print("recursion level: {}".format(level))
    # print("dim. fine system: {}".format(N_fine))
    # print("dim. coarse system: {}".format(N_coarse))
    # print("partition size: {}".format(M))
    # print("halo size: {}".format(n_halo))

    dyn_partition, mtx_cond, mtx_cond_partmax, mtx_cond_partmax_dyn = rptapp_generate_partition(
        a_fine, b_fine, c_fine, M, n_halo, k_max_up, k_max_down)
    N_coarse = len(dyn_partition)*2

    # Reduce to coarse system
    a_coarse = np.zeros(N_coarse)
    b_coarse = np.zeros(N_coarse)
    c_coarse = np.zeros(N_coarse)
    d_coarse = np.zeros(N_coarse)
    
    # TODO: document input/output parameters of rptapp_reduce()
    rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse, d_coarse, 
                  dyn_partition, threshold)
    mtx_coarse = matrix.bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)
    mtx_cond_coarse = np.linalg.cond(mtx_coarse)
    
    # print("\nCoarse system (M = {}, k_max_up = {}, k_max_down = {}), condition: {:e}".format(
    #     M, k_max_up, k_max_down, mtx_cond_coarse))

    if len(a_coarse) <= N_tilde:
        # If system size below threshold, solve directly
        x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
    else:
        # Otherwise, do recursive call
        # Note: each recursive call starts from a static partition, and computes
        # new boundaries based on condition of the partitions
        # TODO: set maximum level for adjusting partition boundaries?
        x_coarse = rptapp(a_coarse, b_coarse, c_coarse, d_coarse, N_tilde, M, 
                          k_max_up, k_max_down, threshold, n_halo, level=level+1)    

    # Substitute into fine system
    x_fine_rptapp = rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, 
                                      dyn_partition, threshold)
    # Return additional values for parameter study
    return x_fine_rptapp, mtx_cond, mtx_cond_coarse, mtx_cond_partmax, mtx_cond_partmax_dyn
  

def rptapp_print(a_fine, b_fine, c_fine, d_fine, x_fine, mtx_id, N_tilde, M, 
                 k_max_up, k_max_down, threshold, n_halo):
    try:
        x_fine_rptapp, cond, cond_coarse, cond_partmax, cond_partmax_dyn = rptapp(
                a_fine, b_fine, c_fine, d_fine, N_tilde, M, k_max_up, k_max_down, 0, n_halo)
        fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
        print("{},{},{},{},{},{},{},{},{}".format(
            mtx_id, M, k_max_up, k_max_down, fre, cond, cond_coarse, cond_partmax, cond_partmax_dyn))

    except np.linalg.LinAlgError:
        print("warning: Singular matrix detected", file=sys.stderr)
        print("{},{},{},{},{},{},{},{},{}".format(
            mtx_id, M, k_max_up, k_max_down, np.Inf, np.Inf, np.Inf, np.Inf, np.Inf))


# %%
N_fine = 512
# M = 61
M = 32
N_tilde = (ceil(N_fine / M)) * 2 # one reduction step
mtx_id = 14
k_max_up = 0
# k_max_up = 1
k_max_down = 0
# k_max_down = 1
n_halo = 1

# Generate fine system
a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
    mtx_id, N_fine, unif_low=-1, unif_high=1, seed=0)

# dyn_partition, mtx_cond, mtx_cond_partmax, mtx_cond_partmax_dyn = rptapp_generate_partition(
#     a_fine, b_fine, c_fine, M, n_halo, k_max_up, k_max_down)
static_partition = condition.generate_static_partition(N_fine, M)
dyn_partition = rptapp_generate_partition_det(a_fine, b_fine, c_fine, 16, 64)

# rpta_partition = dyn_partition
rpta_partition = static_partition
N_coarse = len(rpta_partition)*2

# Reduce to coarse system
a_coarse = np.zeros(N_coarse)
b_coarse = np.zeros(N_coarse)
c_coarse = np.zeros(N_coarse)
d_coarse = np.zeros(N_coarse)

rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse, d_coarse, 
              rpta_partition, threshold=0)
mtx_coarse = matrix.bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)
mtx_cond_coarse = np.linalg.cond(mtx_coarse)

# Plot coarse system
condition.plot_coarse_system(mtx_coarse, "Condition: {:e}".format(mtx_cond_coarse))

# Insert solution
x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
x_fine_rptapp = rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, rpta_partition, threshold=0)
fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)
print("{:e}, {:e}".format(fre, mtx_cond_coarse))

# %%
def main():
    mtx_id = int(sys.argv[1])
    N_fine = int(sys.argv[2])
    n_halo = int(sys.argv[3])
    unif_low, unif_high = -1, 1
    
    # # Solution
    # np.random.seed(0)
    # x_fine = np.random.normal(3, 1, N_fine)
    # # Coefficient matrix
    # mtx = matrix.generate_matrix(mtx_id, N_fine, unif_low, unif_high)
    # a_fine, b_fine, c_fine = matrix.numpy_matrix_to_bands(mtx)
    # # Right-hand side
    # d_fine = np.matmul(mtx, x_fine)

    print("ID,M,k_max_up,k_max_down,fre,cond,cond_coarse,cond_partmax,cond_partmax_dyn")
    a_fine, b_fine, c_fine, d_fine, x_fine = matrix.generate_linear_system(
        mtx_id, N_fine, unif_low, unif_high, 0)

    # Take all combinations of partition size / k_max_up / k_max_down
    Ms = range(16, 65)
    k_sup = 6
    for M in Ms:
        #N_tilde = 64
        N_tilde = (ceil(N_fine / M)) * 2
    
        for k_max_up in range(0, k_sup):
            for k_max_down in range(0, k_sup):
                rptapp_print(a_fine, b_fine, c_fine, d_fine, x_fine, mtx_id, N_tilde, M, 
                             k_max_up, k_max_down, 0, n_halo)

if __name__ == "__main__":
    main()