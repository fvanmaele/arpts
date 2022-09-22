#!/bin/bash
set -ex

# Workload distributed for N cores
parallel --keep-order julia holes.jl 512 {} ::: 11 14 ::: $(seq 8 16) ::: 0.0 1e-16 1e-12 1e-8 1e-4 1.0

# Workload distributed for 6 cores
# e = 0
# parallel --keep-order julia holes.jl 512 {} 0.0 ::: 11 14 ::: 8 12 16

# e = 1e-16
# parallel --keep-order julia holes.jl 512 {} 1e-16 ::: 11 14 ::: 8 12 16

# e = 1e-12
# parallel --keep-order julia holes.jl 512 {} 1e-12 ::: 11 14 ::: 8 12 16

# e = 1e-8
# parallel --keep-order julia holes.jl 512 {} 1e-8 ::: 11 14 ::: 8 12 16

# e = 1e-4
# parallel --keep-order julia holes.jl 512 {} 1e-4 ::: 11 14 ::: 8 12 16

# e = 1 (original system)
# parallel --keep-order julia holes.jl 512 {} 1.0 ::: 11 14 ::: 8 12 16
