#!/bin/bash
set -e
M=32 N=512

cd sol_unifdist
for id in $(seq 1 20); do
    echo "Computing distribution for matrix $id"
    ../distribution.py "$id" "$N" "$M" --seed 0 \
        --rand-n-samples 10000 --n-trials 5000 --distribution 'uniform'
done
cd -

cd sol_normdist
for id in $(seq 1 20); do
    echo "Computing distribution for matrix $id"
    ../distribution.py "$id" "$N" "$M" --seed 0 \
        --rand-n-samples 10000 --n-trials 5000 --distribution 'normal'
done
