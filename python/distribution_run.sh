#!/bin/bash
set -e
M=32 N=512 n_trials=1000
distribution=${1:-normal}
distribution_setup=${2:-normal}

for id in $(seq 1 25); do
    printf 'Computing distribution for matrix %d, solutions ~ %s, generated from ~ %s' "$id" "$distribution" "$distribution_setup"
    ./distribution.py "$id" "$N" "$M" --seed 0 \
                      --rand-n-samples 5000 --n-trials "$n_trials" \
                      --distribution "$distribution" \
                      --distribution-setup "$distribution_setup"
done
