#!/bin/bash
set -ex

# Workload distributed for N cores, threads
for id in 11 14; do
    for n_holes in $(seq 8 16); do
        for eps in 0.0 1e-16 1e-12 1e-8 1e-4 1e-3 1e-2 1e-1; do
            # generate linear system (.mtx, .json)
            julia --threads=24 holes.jl 512 "$id" "$n_holes" "$eps"
            # move files to subdirectories
            python3 sort_files.py
        done
    done
done

# Workload distributed for N cores, external parallelism
# parallel --keep-order julia holes.jl 512 {} ::: 11 14 ::: $(seq 8 16) ::: 0.0 1e-16 1e-12 1e-8 1e-4 1e-3 1e-2 1e-1
