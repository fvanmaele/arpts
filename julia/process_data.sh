#!/bin/bash
set -x

# find . -mindepth 3 -type d | sort -u
dirs=(
    ./decoupled_eps_0.00e+00/11/08
    ./decoupled_eps_0.00e+00/11/12
    ./decoupled_eps_0.00e+00/11/16
    ./decoupled_eps_0.00e+00/14/08
    ./decoupled_eps_0.00e+00/14/12
    ./decoupled_eps_0.00e+00/14/16
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
    ./decoupled_eps_1.00e-03/11/12
    ./decoupled_eps_1.00e-03/11/16
    ./decoupled_eps_1.00e-03/14/08
    ./decoupled_eps_1.00e-03/14/12
    ./decoupled_eps_1.00e-03/14/16
    ./decoupled_eps_1.00e-04/11/08
    ./decoupled_eps_1.00e-04/11/12
    ./decoupled_eps_1.00e-04/11/16
    ./decoupled_eps_1.00e-04/14/08
    ./decoupled_eps_1.00e-04/14/12
    ./decoupled_eps_1.00e-04/14/16
    ./decoupled_eps_1.00e-08/11/08
    ./decoupled_eps_1.00e-08/11/12
    ./decoupled_eps_1.00e-08/11/16
    ./decoupled_eps_1.00e-08/14/08
    ./decoupled_eps_1.00e-08/14/12
    ./decoupled_eps_1.00e-08/14/16
    ./decoupled_eps_1.00e-12/11/08
    ./decoupled_eps_1.00e-12/11/12
    ./decoupled_eps_1.00e-12/11/16
    ./decoupled_eps_1.00e-12/14/08
    ./decoupled_eps_1.00e-12/14/12
    ./decoupled_eps_1.00e-12/14/16
    ./decoupled_eps_1.00e-16/11/08
    ./decoupled_eps_1.00e-16/11/12
    ./decoupled_eps_1.00e-16/11/16
    ./decoupled_eps_1.00e-16/14/08
    ./decoupled_eps_1.00e-16/14/12
    ./decoupled_eps_1.00e-16/14/16
)

# Run on N cores
parallel --keep-order --halt now,fail=1 python process_data.py {} ::: "${dirs[@]}"

#for d in "${dirs[@]}"; do
#    python process_data.py "$d" || exit 1
#done
