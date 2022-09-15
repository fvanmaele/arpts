#!/usr/bin/julia
using JSON
using LinearAlgebra
using MatrixMarket
using SparseArrays
using Random
using Printf

# The difference to `generate_random_partition` in python/partition.py is that
# all samples have the same amount of partitions (no merging of partition boundaries)
function generate_holes(rng, n_last, n_holes, n_min_part, n_max_part, n_samples; 
                        n_max_attempts=5000)
    holes = Vector{Vector{Int64}}()
    n_holes_done = 0
    attempts = 0

    while n_holes_done < n_samples
        is_valid = true  # determines if a partition is of a given size
        sample = sort(vcat([1; randperm(N)[1:n_holes]; n_last])) # sorted array of length n_holes+2
        
        # TODO: if last partition is too low, merge into upper neighbor
        for part in Iterators.partition(sample, 2)
            part_size = part[2] - part[1]
    
            if !(part_size <= n_part_max_size && part_size >= n_part_min_size)
                is_valid = false
            end
        end

        if is_valid
            # copy because we reassign `sample` in every iteration
            push!(holes, convert(Vector{Int64}, sample[2:n_holes+1]))
            n_holes_done += 1
        end

        attempts += 1
        if attempts > n_max_attempts
            error("maximum number of attempts exceeded, valid samples: " * string(n_holes_done))
        end
    end
    return holes
end

# Solution for a fixed right-hand side `b` and variable coefficient matrix `A`, 
# computed in multiple precision. An upper bound on the accuracy is computed.
function tridiag_exact_solution(T::AbstractMatrix{Float64}, rhs::AbstractVector{Float64})
    # extended precision matrix inverse
    # note: the inverse of a (sparse) tridiagional matrix is dense in general.
    Tu = Tridiagonal(diag(T, -1), diag(T), diag(T, 1))
    Tu = convert(Tridiagonal{BigFloat}, Tu)
    Tu_inv = inv(Tu)

    # compute corresponding solution
    sol = Tu_inv * rhs
    
    # sanity check: upper bound on forward relative error
    res = norm(Tu * sol - rhs)  # residual
    acc = norm(Tu_inv) * (res / norm(sol))
    
    return sol, res, acc
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
holes = generate_holes(rng, N, n_holes, n_part_min_size, n_part_max_size, n_holes_samples; 
                       n_max_attempts=500000)
@assert sum(unique(map(length, holes))) == n_holes

# set rows determined by hole indices to 0 (later: small epsilon)
# S_samples = Vector{SparseMatrixCSC}()
for mtx_id in idx
    S = MatrixMarket.mmread("mtx/" * string(mtx_id) * "-" * string(N) * ".mtx")
    S_dl, S_d, S_du = diag(S, -1), diag(S), diag(S, 1)
    
    Threads.@threads for k in 1:n_holes_samples
        sample = holes[k]
        dl, d, du = copy(S_dl), copy(S_d), copy(S_du)
        dl[sample] .= 0.0
        du[sample] .= 0.0

        # push!(S_samples, dropzeros(SparseMatrixCSC(Tridiagonal(dl, d, du))))
        fname = @sprintf("mtx-%i-%i-decoupled-%04i.mtx", mtx_id, N, k)
        S_new = dropzeros(SparseMatrixCSC(Tridiagonal(dl, d, du)))
        S_new_cond = cond(Array(S_new), 2)
        MatrixMarket.mmwrite(fname, S_new)
    
        jname = @sprintf("mtx-%i-%i-decoupled-%04i.json", mtx_id, N, k) # 1-indexed positions
        sol, res, acc = tridiag_exact_solution(S_new, rhs)
        open(jname, "w") do f
            JSON.print(f, Dict("sample_1idx" => sample, "solution" => sol, "max_accuracy" => acc, "residual" => res, "condition" => S_new_cond))
        end
    end
end
