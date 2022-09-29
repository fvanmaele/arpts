#!/bin/bash
set -x

# find . -mindepth 3 -type d | sort -u
dirs=(
#    ./decoupled_eps_0.00e+00/mtx_11/08
#    ./decoupled_eps_0.00e+00/mtx_11/12
#    ./decoupled_eps_0.00e+00/mtx_11/16
#    ./decoupled_eps_0.00e+00/mtx_14/08
#    ./decoupled_eps_0.00e+00/mtx_14/12
#    ./decoupled_eps_0.00e+00/mtx_14/16
#    ./decoupled_eps_1.00e+00/mtx_11/08
#    ./decoupled_eps_1.00e+00/mtx_11/12
#    ./decoupled_eps_1.00e+00/mtx_11/16
#    ./decoupled_eps_1.00e+00/mtx_14/08
#    ./decoupled_eps_1.00e+00/mtx_14/12
#    ./decoupled_eps_1.00e+00/mtx_14/16
    ./decoupled_eps_1.00e-01/11/08
    ./decoupled_eps_1.00e-01/11/12
    ./decoupled_eps_1.00e-01/11/16
    ./decoupled_eps_1.00e-01/14/08
    ./decoupled_eps_1.00e-01/14/12
    ./decoupled_eps_1.00e-01/14/16
    ./decoupled_eps_1.00e-02/11/08
    ./decoupled_eps_1.00e-02/11/12
    ./decoupled_eps_1.00e-02/11/16
    ./decoupled_eps_1.00e-02/14/08
    ./decoupled_eps_1.00e-02/14/12
    ./decoupled_eps_1.00e-02/14/16
    ./decoupled_eps_1.00e-03/11/08
#    ./decoupled_eps_1.00e-04/mtx_11/08
#    ./decoupled_eps_1.00e-04/mtx_11/12
#     ./decoupled_eps_1.00e-04/mtx_11/16
#     ./decoupled_eps_1.00e-04/mtx_14/08
#     ./decoupled_eps_1.00e-04/mtx_14/12
#     ./decoupled_eps_1.00e-04/mtx_14/16
#     ./decoupled_eps_1.00e-08/mtx_11/08
#     ./decoupled_eps_1.00e-08/mtx_11/12
#     ./decoupled_eps_1.00e-08/mtx_11/16
#     ./decoupled_eps_1.00e-08/mtx_14/08
#     ./decoupled_eps_1.00e-08/mtx_14/12
#     ./decoupled_eps_1.00e-08/mtx_14/16
#     ./decoupled_eps_1.00e-12/mtx_11/08
#     ./decoupled_eps_1.00e-12/mtx_11/12
#     ./decoupled_eps_1.00e-12/mtx_11/16
#     ./decoupled_eps_1.00e-12/mtx_14/08
#     ./decoupled_eps_1.00e-12/mtx_14/12
#     ./decoupled_eps_1.00e-12/mtx_14/16
#     ./decoupled_eps_1.00e-16/mtx_11/08
#     ./decoupled_eps_1.00e-16/mtx_11/12
#     ./decoupled_eps_1.00e-16/mtx_11/16
#     ./decoupled_eps_1.00e-16/mtx_14/08
#     ./decoupled_eps_1.00e-16/mtx_14/12
#     ./decoupled_eps_1.00e-16/mtx_14/16
)

# Run on N cores
parallel --keep-order --halt now,fail=1 python process_data.py {} ::: "${dirs[@]}"

#for d in "${dirs[@]}"; do
#    python process_data.py "$d" || exit 1
#done
