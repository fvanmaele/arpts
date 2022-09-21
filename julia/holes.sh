#!/bin/bash
# Workload distributed for 6 cores
set -e

# e = 0
parallel --keep-order julia holes.jl 512 {} {} 0.0 ::: 11 14 ::: 8 12 16
# julia holes.jl 512 11 8 0.0
# julia holes.jl 512 11 12 0.0
# julia holes.jl 512 11 16 0.0
# julia holes.jl 512 14 8 0.0
# julia holes.jl 512 14 12 0.0
# julia holes.jl 512 14 16 0.0

# e = 1e-16
parallel --keep-order julia holes.jl 512 {} {} 1e-16 ::: 11 14 ::: 8 12 16
# julia holes.jl 512 11 8 1e-16
# julia holes.jl 512 11 12 1e-16
# julia holes.jl 512 11 16 1e-16
# julia holes.jl 512 14 8 1e-16
# julia holes.jl 512 14 12 1e-16
# julia holes.jl 512 14 16 1e-16

# e = 1e-12
parallel --keep-order julia holes.jl 512 {} {} 1e-12 ::: 11 14 ::: 8 12 16
# julia holes.jl 512 11 8 1e-12
# julia holes.jl 512 11 12 1e-12
# julia holes.jl 512 11 16 1e-12
# julia holes.jl 512 14 8 1e-12
# julia holes.jl 512 14 12 1e-12
# julia holes.jl 512 14 16 1e-12

# e = 1e-8
parallel --keep-order julia holes.jl 512 {} {} 1e-8 ::: 11 14 ::: 8 12 16
# julia holes.jl 512 11 8 1e-8
# julia holes.jl 512 11 12 1e-8
# julia holes.jl 512 11 16 1e-8
# julia holes.jl 512 14 8 1e-8
# julia holes.jl 512 14 12 1e-8
# julia holes.jl 512 14 16 1e-8

# e = 1e-4
parallel --keep-order julia holes.jl 512 {} {} 1e-4 ::: 11 14 ::: 8 12 16
# julia holes.jl 512 11 8 1e-4
# julia holes.jl 512 11 12 1e-4
# julia holes.jl 512 11 16 1e-4
# julia holes.jl 512 14 8 1e-4
# julia holes.jl 512 14 12 1e-4
# julia holes.jl 512 14 16 1e-4

# e = 1 (original system)
parallel --keep-order julia holes.jl 512 {} {} 1.0 ::: 11 14 ::: 8 12 16
# julia holes.jl 512 11 8 1.0
# julia holes.jl 512 11 12 1.0
# julia holes.jl 512 11 16 1.0
# julia holes.jl 512 14 8 1.0
# julia holes.jl 512 14 12 1.0
# julia holes.jl 512 14 16 1.0
