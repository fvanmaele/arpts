#!/bin/bash
set -xeo pipefail
N=512
n_halo=0
#M=($(seq 16 64))
#k_up_lim=5
#k_down_lim=5

parallel --quote bash -c "./rpta.py {} $N $n_halo | tee {}_N${N}_M16-64_u5_d5_halo${n_halo}.csv" ::: {01..20}

