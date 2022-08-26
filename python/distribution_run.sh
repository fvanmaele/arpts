#!/bin/bash
set -e
M=32 N=512 n_trials=500
distribution=${1:-normal}
distribution_setup=${2:-normal}

mtx=('11-512-1e+04'
     '11-512-1e+05'
     '11-512-1e+06'
     '11-512-1e+07'
     '11-512-1e+08'
     '11-512-1e+09'
     '11-512-1e+10'
     '11-512-1e+11'
     '11-512-1e+12'
     '14-512-0e+00'
     '14-512-1e-01'
     '14-512-1e-02'
     '14-512-1e-03'
     '14-512-1e-04'
     '14-512-1e-05'
     '14-512-1e-06'
     '14-512-1e-07'
     '14-512-1e-08')

for id in "${mtx[@]}"; do
    printf 'Computing distribution for matrix %s, solutions ~ %s, generated from ~ %s\n' "$id" "$distribution" "$distribution_setup"
    python distribution.py "$id" "$N" "$M" --seed 0 --n-trials "$n_trials" \
                      --rand-n-samples 2000 --rand-min-part 24 --rand-max-part 48 --rand-mean 32 --rand-sd 2 \
                      --static-M-min 16 --static-M-max 64 --static-min-part 16 \
                      --cond-lo-min 16 --cond-lo-max 48 --cond-hi-min 32 --cond-hi-max 64 --cond-min-part 16 \
                      --distribution "$distribution" \
                      --distribution-setup "$distribution_setup"
done
