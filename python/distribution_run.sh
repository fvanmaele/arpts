#!/bin/bash
set -e
M=32 N=512 n_trials=1000
distribution=${1:-normal}
distribution_setup=${2:-normal}

for id in $(seq 1 30); do
    printf 'Computing distribution for matrix %d, solutions ~ %s, generated from ~ %s\n' "$id" "$distribution" "$distribution_setup"
    ./distribution.py "$id" "$N" "$M" --seed 0 \
                      --rand-n-samples 2000 --rand-min-part 16 --rand-max-part 64 --rand-mean 32 --rand-sd 2 \
                      --static-M-min 16 --static-M-max 64 \
                      --n-trials "$n_trials" \
                      --distribution "$distribution" \
                      --distribution-setup "$distribution_setup" \
                      --symmetric
done
