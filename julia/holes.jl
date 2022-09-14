#!/usr/bin/julia
using JSON
using LinearAlgebra
using SparseArrays
using MatrixMarket
using Random

function generate_holes(rng, n_holes, n_min_part, n_max_part, n_samples; 
                        n_max_attempts=5000)
    holes = Vector{Vector{Float64}}()
    n_holes_done = 0
    attempts = 0

    while n_holes_done < n_holes_samples
        is_valid = true
        sample = sort(randperm(N)[1:n_holes])
        
        for part in Iterators.partition(sample, 2)
            part_size = part[2] - part[1]
    
            if !(part_size <= n_part_max_size && part_size >= n_part_min_size)
                is_valid = false
            end
        end

        if is_valid
            # copy because we reassign `sample` in every iteration
            push!(holes, copy(sample))
            n_holes_done += 1
        end

        attempts += 1
        if attempts > n_max_attempts
            error("maximum number of attempts exceeded, valid samples: " * string(n_holes_done))
        end
    end
    return holes
end

# linear system with fixed right-hand side
N = 512
idx = [11, 14]
rhs = ones(N)

# generate "holes" in matrix
#n_holes_min = 8  # corresponding to partition size of M=64
#n_holes_max = 16 # corresponding to partition size of M=32
n_holes = 16
n_part_min_size = 16
n_part_max_size = 80
n_holes_samples = 1000 # number of samples within partition bounds
n_holes_max_attempts = 5000

# holes are generated independent of the considered matrix
rng = MersenneTwister(1234)
holes = generate_holes(rng, n_holes, n_part_min_size, n_part_max_size, n_holes_samples; n_max_attempts=500000)

# holes = Vector{Vector{Float64}}()
# n_holes_done = 0

for id in idx
    S = MatrixMarket.mmread("mtx/" * id * ".mtx")
    S = Tridiagonal(diag(S, -1), diag(S), diag(S, 1))

end
