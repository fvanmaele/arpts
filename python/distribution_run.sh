#!/bin/bash
set -e
M=32 N=512

for id in $(seq 1 20); do
    echo "Computing distribution for matrix $id"
    ./distribution.py "$id" "$N" "$M" --seed 0 --n-trials 5000 --distribution uniform
done
