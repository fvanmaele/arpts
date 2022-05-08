#!/bin/bash
set -eo pipefail
#N=2048
N=512
# TODO: (maybe in other script): save partition with minimal FRE, do
# trials over 1000 sampled right-hand sides
# TODO: do some statistical analysis on dependence of FRE on
# hyperparameters (e.g.  main_cond_coarse.py has less dependence on
# lim_lo, lim_hi than main_rows.py)

# Static partition, M = 16..64
{ echo ID,N,M,fre,cond_coarse
  for id in $(seq 1 20); do
      ./main_static.py "$id" "$N" 16-64 --min-size 5
  done
} | tee "output_${N}_static.csv"

# Generate coarse system of minimal condition during reduction step
{ echo ID,N,lim_lo,lim_hi,fre,cond_coarse
  for id in $(seq 1 20); do
      ./main_cond_coarse.py "$id" "$N" 16-40 22-72 --min-size=6
  done
} | tee "output_${N}_min_cond_coarse.csv"

# Generate random partition
{ echo ID,N,fre,cond_coarse
  for id in $(seq 1 20); do
      ./main_random.py "$id" "$N" 5000 32 100
  done
} | tee "output_${N}_random_32_100.csv"

# Check rows of linear system for linear independence [using determinant]
{ echo ID,N,lim_lo,lim_hi,fre,cond_coarse
  for id in $(seq 1 20); do
      ./main_rows.py "$id" "$N" 16-40 22-72 --min-size=6 cond
  done
} | tee "output_${N}_rows_cond.csv"

# Check rows of linear system for linear independence [using condition]
{ echo ID,N,lim_lo,lim_hi,fre,cond_coarse
  for id in $(seq 1 20); do
      ./main_rows.py "$id" "$N" 16-40 22-72 --min-size=6 det
  done
} | tee "output_${N}_rows_det.csv"

# Dynamic partition, minimize condition of individual blocks
# TODO: check negative k_max_{up,down}
# TODO: include halo in the csv output
for n_halo in $(seq 0 2); do
    { echo ID,M,k_max_up,k_max_down,fre,cond,cond_coarse,cond_partmax,cond_partmax_dyn
      for id in $(seq 1 20); do
          ./main_cond_part.py "$id" "$N" 16-64 0-5 0-5 "$n_halo"
      done
    } | tee "output_${N}_min_cond_part_halo${n_halo}".csv
done
