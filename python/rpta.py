#!/usr/bin/env python
# This code is based on an implementation by Christoph Klein.
# Reference paper: "Tridiagonal GPU Solver with Scaled Partial Pivoting
# at Maximum Bandwidth, C. Klein, R. Strzodka, ICPP '21"

import numpy as np
import sys
import matrix # bands_to_numpy_matrix

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


# TODO: add visualization
def eliminate_band(a, b, c, d, pivoting, threshold=0):
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

        # Scaled partial pivoting
        if pivoting == "scaled_partial":
            m_p = max([abs(s_p[0]), abs(s_p[1]), abs(s_p[2])])
            m_c = max([abs(s_c[1]), abs(s_c[2]), abs(s_c[3])])
        elif pivoting == "partial":
            m_p = 1.0
            m_c = 1.0
        elif pivoting == "none":
            m_p = 0.0
            m_c = 0.0

        if abs(s_c[1])*m_p > abs(s_p[1])*m_c:
            # print(s_c[1])
            r_c = (-1.0) * s_p[1] / s_c[1]
            r_p = 1.0
        else:
            # print(s_p[1])
            r_c = 1.0
            r_p = (-1.0) * s_c[1] / s_p[1]

        for k in [0, 2, 3, 4]:
            s_p[k] = r_p * s_p[k] + r_c * s_c[k]

        s_p[1] = s_p[2]
        s_p[2] = s_p[3]
        s_p[3] = 0.0

    return s_p[0], s_p[1], s_p[2], s_p[4]


# TODO: adjust for partitions with dynamic boundaries (take list as input)
def rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse, d_coarse, 
                  partition, pivoting, threshold=0):
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
            pivoting, threshold)
        c_coarse_upper, b_coarse_upper, a_coarse_upper, d_coarse_upper = eliminate_band(
            list(reversed(c_fine[partition_begin:partition_end])),
            list(reversed(b_fine[partition_begin:partition_end])),
            list(reversed(a_fine[partition_begin:partition_end])),
            list(reversed(d_fine[partition_begin:partition_end])),
            pivoting, threshold)

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
def eliminate_band_with_solution(a, b, c, d, x1_prev_partition, x0, x1, x0_next_partition, 
                                 pivoting, threshold=0):
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

        # Scaled partial pivoting
        if pivoting == "scaled_partial":
            m_p = max([abs(s_p[1]), abs(s_p[2])])
            m_c = max([abs(s_c[1]), abs(s_c[2]), abs(s_c[3])])
        elif pivoting == "partial":
            m_p = 1.0
            m_c = 1.0
        elif pivoting == "none":
            m_p = 0.0
            m_c = 0.0

        if abs(s_c[1])*m_p > abs(s_p[1])*m_c:
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

    # FIXME: "None" k in last partition of size N % M
    if k is not None and abs(a[k]) < abs(c[0]):
        x[1] = (d[0] - a[0] * x1_prev_partition - b[0] * x[0]) / c[0]
    elif k is not None:
        x[1] = (d[k] - b[k] * x[2] - c[k] * x[3]) / a[k]

    return x


def rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, partition, 
                      pivoting, threshold=0):
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
            x1_prev_partition, x0, x1, x0_next_partition, 
            pivoting, threshold)

        x_fine[partition_begin:partition_end] = x_partition

    return x_fine


# TODO: keep partial results (in eleminate_band) of coarse system and compute determinant 
# The observation here is that consecutive rows in the coarse system are, for a bad partitioning,
# nearly linearly independent. As such, we can optimize for this when constructing (rows of)
# the coarse system.
def rptapp_reduce_dynamic(a_fine, b_fine, c_fine, d_fine, part_min, part_max, 
                          pivoting='scaled_partial'):
    N = len(a_fine)
    assert(part_min < part_max)
    assert(part_min > 0)
    assert(part_max < N-1)
    
    partition = []
    partition_begin = 0 # Index of first row (upper boundary) 
    
    while partition_begin + part_max < N:
        conds = []
        for offset in range(part_min, part_max):
            partition_end = min(partition_begin + offset, N-1)
            
            a_coarse_lower, b_coarse_lower, c_coarse_lower, d_coarse_lower = eliminate_band(
                a_fine[partition_begin:partition_end],
                b_fine[partition_begin:partition_end],
                c_fine[partition_begin:partition_end],
                d_fine[partition_begin:partition_end],
                pivoting, threshold=0)
            c_coarse_upper, b_coarse_upper, a_coarse_upper, d_coarse_upper = eliminate_band(
                list(reversed(c_fine[partition_begin:partition_end])),
                list(reversed(b_fine[partition_begin:partition_end])),
                list(reversed(a_fine[partition_begin:partition_end])),
                list(reversed(d_fine[partition_begin:partition_end])),
                pivoting, threshold=0)
            
            # Criterion: minimum condition
            mtx = np.matrix([[b_coarse_upper, c_coarse_upper],
                             [a_coarse_lower, b_coarse_lower]])
            mtx_cond = np.linalg.cond(mtx)
            conds.append(mtx_cond)
            # print("{}, {}: |det| = {}".format(partition_begin, partition_end, mtx_det))

        conds_argmin = np.argmin(conds)
        # print("{}, {}: |det| (max) = {}".format(
        #     partition_begin, partition_begin + part_min + dets_argmax, dets[dets_argmax]))
        partition.append([partition_begin, partition_begin + part_min + conds_argmin])

        # Go to next partition
        partition_begin = min(partition_begin + part_min + conds_argmin, N)
    
    # Append last partition
    if partition_begin < N:
        partition.append([partition_begin, N])

    return partition


# TODO: support recursion
def reduce_and_solve(N_coarse, a_fine, b_fine, c_fine, d_fine, partition, 
                     pivoting='scaled_partial'):
    # Reduce to coarse system
    a_coarse = np.zeros(N_coarse)
    b_coarse = np.zeros(N_coarse)
    c_coarse = np.zeros(N_coarse)
    d_coarse = np.zeros(N_coarse)
    
    rptapp_reduce(a_fine, b_fine, c_fine, d_fine, a_coarse, b_coarse, c_coarse, d_coarse,
                  partition, pivoting, threshold=0)
    mtx_coarse = matrix.bands_to_numpy_matrix(a_coarse, b_coarse, c_coarse)

    # Plot coarse system
    # partition.plot_coarse_system(mtx_coarse, "Condition: {:e}".format(mtx_cond_coarse))
    try:
        x_coarse = np.linalg.solve(mtx_coarse, d_coarse)
        x_fine_rptapp = rptapp_substitute(a_fine, b_fine, c_fine, d_fine, x_coarse, 
                                          partition, pivoting, threshold=0)
        mtx_cond_coarse = np.linalg.cond(mtx_coarse)
        # fre = np.linalg.norm(x_fine_rptapp - x_fine) / np.linalg.norm(x_fine)

    except np.linalg.LinAlgError:
        print("warning: Singular matrix detected", file=sys.stderr)
        x_fine_rptapp, mtx_coarse, mtx_cond_coarse = None, None, np.Inf
        
    return x_fine_rptapp, mtx_coarse, mtx_cond_coarse
