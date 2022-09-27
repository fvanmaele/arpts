#!/bin/bash
set -ex
dirs=(./decoupled_eps_1.00e-12/mtx_14/08
      ./decoupled_eps_1.00e-12/mtx_14/16
      ./decoupled_eps_1.00e-12/mtx_14/12
      ./decoupled_eps_1.00e-12/mtx_11/08
      ./decoupled_eps_1.00e-12/mtx_11/16
      ./decoupled_eps_1.00e-12/mtx_11/12
      ./decoupled_eps_1.00e+00/mtx_14/08
      ./decoupled_eps_1.00e+00/mtx_14/16
      ./decoupled_eps_1.00e+00/mtx_14/12
      ./decoupled_eps_1.00e+00/mtx_11/08
      ./decoupled_eps_1.00e+00/mtx_11/16
      ./decoupled_eps_1.00e+00/mtx_11/12
      ./decoupled_eps_1.00e-04/mtx_14/08
      ./decoupled_eps_1.00e-04/mtx_14/16
      ./decoupled_eps_1.00e-04/mtx_14/12
      ./decoupled_eps_1.00e-04/mtx_11/08
      ./decoupled_eps_1.00e-04/mtx_11/16
      ./decoupled_eps_1.00e-04/mtx_11/12
      ./decoupled_eps_1.00e-08/mtx_14/08
      ./decoupled_eps_1.00e-08/mtx_14/16
      ./decoupled_eps_1.00e-08/mtx_14/12
      ./decoupled_eps_1.00e-08/mtx_11/08
      ./decoupled_eps_1.00e-08/mtx_11/16
      ./decoupled_eps_1.00e-08/mtx_11/12
      ./decoupled_eps_1.00e-16/mtx_14/08
      ./decoupled_eps_1.00e-16/mtx_14/16
      ./decoupled_eps_1.00e-16/mtx_14/12
      ./decoupled_eps_1.00e-16/mtx_11/08
      ./decoupled_eps_1.00e-16/mtx_11/16
      ./decoupled_eps_1.00e-16/mtx_11/12
      ./decoupled_eps_0.00e+00/mtx_14/08
      ./decoupled_eps_0.00e+00/mtx_14/16
      ./decoupled_eps_0.00e+00/mtx_14/12
      ./decoupled_eps_0.00e+00/mtx_11/08
      ./decoupled_eps_0.00e+00/mtx_11/16
      ./decoupled_eps_0.00e+00/mtx_11/12
     )

#parallel --keep-order python process_data.py {} ::: "${dirs[@]}"

for d in "${dirs[@]}"; do
    python process_data.py "$d" || exit 1
done
