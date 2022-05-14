#!/bin/bash
set -e
M=32 N=512

# TODO: specify target directory for .png to distribution.py
for id in $(seq 16 20); do
    echo "Computing distribution for matrix $id, solutions ~ normal, generated from ~ normal"
    ./distribution.py "$id" "$N" "$M" --seed 0 \
        --rand-n-samples 5000 --n-trials 5000 --distribution normal --distribution-setup normal
done
mv ecdf*.png part_normdist/sol_normdist

for id in $(seq 1 20); do
    echo "Computing distribution for matrix $id, solutions ~ uniform, generated from ~ normal"
    ./distribution.py "$id" "$N" "$M" --seed 0 \
        --rand-n-samples 5000 --n-trials 5000 --distribution uniform --distribution-setup normal
done
mv ecdf*.png part_normdist/sol_unifdist

for id in $(seq 1 20); do
    echo "Computing distribution for matrix $id, solutions ~ normal, generated from ~ uniform"
    ./distribution.py "$id" "$N" "$M" --seed 0 \
        --rand-n-samples 5000 --n-trials 5000 --distribution normal --distribution-setup uniform
done
mv ecdf*.png part_unifdist/sol_normdist

for id in $(seq 1 20); do
    echo "Computing distribution for matrix $id, solutions ~ uniform, generated from ~ uniform"
    ./distribution.py "$id" "$N" "$M" --seed 0 \
        --rand-n-samples 5000 --n-trials 5000 --distribution uniform --distribution-setup uniform
done
mv ecdf*.png part_unifdist/sol_unifdist
