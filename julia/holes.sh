#!/bin/bash
set -e

# e = 0
julia holes.jl 512 11 8 0.0
julia holes.jl 512 11 12 0.0
julia holes.jl 512 11 16 0.0
julia holes.jl 512 14 8 0.0
julia holes.jl 512 14 12 0.0
julia holes.jl 512 14 16 0.0

# e = 1e-16

# e = 1e-12

# e = 1e-8
