#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 17:11:15 2022

@author: archie
"""

# %%
def eliminate_band_iter(a, b, c, d, pivoting='scaled_partial'):
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

    yield s_p

    for j in range(2, M):
        s_c[0] = 0.0
        s_c[1] = a[j]
        s_c[2] = b[j]
        s_c[3] = c[j]
        s_c[4] = d[j]

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
            # print("{} * {} / {} (pivoted)".format(-1, s_p[1], s_c[1]))
            r_c = (-1.0) * s_p[1] / s_c[1]
            r_p = 1.0
        else:
            # print("{} * {} / {}".format(-1, s_c[1], s_p[1]))
            r_c = 1.0
            r_p = (-1.0) * s_c[1] / s_p[1]

        for k in [0, 2, 3, 4]:
            # print("{} * {} + {} * {}".format(r_p, s_p[k], r_c, s_c[k]))
            s_p[k] = r_p * s_p[k] + r_c * s_c[k]

        s_p[1] = s_p[2]
        s_p[2] = s_p[3]
        s_p[3] = 0.0

        yield s_p


def eliminate_band_iter_reversed(a, b, c, d, pivoting='scaled_partial'):
    a_rev = list(reversed(a))
    b_rev = list(reversed(b))
    c_rev = list(reversed(c))
    d_rev = list(reversed(d))
    
    yield from eliminate_band_iter(c_rev, b_rev, a_rev, d_rev, pivoting)


    
# %%
def rpta_symmetric(a, b, c, d, partition, pivoting='scaled_partial'):
    M = len(a)
    assert(M > 1) # band should at least have one element

    # downwards elimination
    b_down = [0.0] * M
    c_down = [0.0] * M
    d_down = [0.0] * M
    s_down = [0.0] * M  # spike

    for j, s_p in enumerate(eliminate_band_iter(a, b, c, d, pivoting), start=1):
        s_down[j] = s_p[0]
        b_down[j] = s_p[1]
        c_down[j] = s_p[2]
        d_down[j] = s_p[4]
        
    # upwards elimination
    a_up = [0.0] * M
    b_up = [0.0] * M
    d_up = [0.0] * M
    s_up = [0.0] * M # spike
    
    for j, s_r in enumerate(eliminate_band_iter_reversed(a, b, c, d, pivoting), start=1):
        s_up[j] = s_r[0]
        b_up[j] = s_r[1]
        a_up[j] = s_r[2]
        d_up[j] = s_r[4]

    # TODO: solve 2x2 linear systems
    return (b_down, c_down, d_down, s_down, 
            list(reversed(a_up)), list(reversed(b_up)), list(reversed(d_up)), list(reversed(s_up)))