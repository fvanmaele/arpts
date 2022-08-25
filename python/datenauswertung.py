#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  1 21:35:12 2022

@author: fvanmaele
"""
import json
import numpy as np
import matplotlib.pyplot as plt

# %%
trials = [
    # 'trials_11-512-1e+04_512_sol-norm_gen-norm.json',
    # 'trials_11-512-1e+05_512_sol-norm_gen-norm.json',
    # 'trials_11-512-1e+06_512_sol-norm_gen-norm.json',
    # 'trials_11-512-1e+07_512_sol-norm_gen-norm.json',
    'trials_11-512-1e+08_512_sol-norm_gen-norm.json',
    'trials_11-512-1e+09_512_sol-norm_gen-norm.json',
    'trials_11-512-1e+10_512_sol-norm_gen-norm.json',
    'trials_11-512-1e+11_512_sol-norm_gen-norm.json',
    # 'trials_11-512-1e+12_512_sol-norm_gen-norm.json',
    # 'trials_14-512-0e+00_512_sol-norm_gen-norm.json',
    # 'trials_14-512-1e-01_512_sol-norm_gen-norm.json',
    # 'trials_14-512-1e-02_512_sol-norm_gen-norm.json',
    # 'trials_14-512-1e-03_512_sol-norm_gen-norm.json',
    # 'trials_14-512-1e-04_512_sol-norm_gen-norm.json',
    'trials_14-512-1e-05_512_sol-norm_gen-norm.json',
    'trials_14-512-1e-06_512_sol-norm_gen-norm.json',
    'trials_14-512-1e-07_512_sol-norm_gen-norm.json',
    'trials_14-512-1e-08_512_sol-norm_gen-norm.json'
]

# %%
trials_dict = {}
for T in trials:  # ordered
    T_dict = {}
    with open(T) as f:
        data = json.load(f)
        for key in data:
            errors = data[key]
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            T_dict[key] = [mean_error, std_error]
    trials_dict[T] = T_dict

# %% label dict
foo = {
       'rand_min_cond_norm': 'I',
       'rand_min_cond_unif': 'II',
       'reduce_min_cond': 'III',
       'static_M32': 'IV',
       'static_min_cond': 'V'
       }

# %%
images = []
for T in sorted(trials_dict.keys()):
    x = []
    y = []
    e = []
    # print(trials_dict[T])
    for key in sorted(trials_dict[T].keys()):
        x.append(foo[key])
        y.append(trials_dict[T][key][0])
        e.append(trials_dict[T][key][1])
    
    images.append([x, y, e])
    
# %%
fig, axs = plt.subplots(4, 2)
axs[0, 0].errorbar(images[0][0], images[0][1], images[0][2], linestyle='None', marker='o')
axs[0, 0].set_title('11-512-1e+08')
axs[0, 1].errorbar(images[1][0], images[1][1], images[1][2], linestyle='None', marker='o')
axs[0, 1].set_title('11-512-1e+09')
axs[1, 0].errorbar(images[2][0], images[2][1], images[2][2], linestyle='None', marker='o')
axs[1, 0].set_title('11-512-1e+10')
axs[1, 1].errorbar(images[3][0], images[3][1], images[3][2], linestyle='None', marker='o')
axs[1, 1].set_title('11-512-1e+11')

axs[2, 0].errorbar(images[4][0], images[4][1], images[4][2], linestyle='None', marker='o')
axs[2, 0].set_title('14-512-1e-05')
axs[2, 1].errorbar(images[5][0], images[5][1], images[5][2], linestyle='None', marker='o')
axs[2, 1].set_title('14-512-1e-06')
axs[3, 0].errorbar(images[6][0], images[6][1], images[6][2], linestyle='None', marker='o')
axs[3, 0].set_title('14-512-1e-07')
axs[3, 1].errorbar(images[7][0], images[7][1], images[7][2], linestyle='None', marker='o')
axs[3, 1].set_title('14-512-1e-08')

for ax in axs.flat:
    ax.set(xlabel='Case', ylabel='FRE')

# Hide x labels and tick labels for top plots and y ticks for right plots.
for ax in axs.flat:
    ax.label_outer()
