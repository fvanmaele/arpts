#!/usr/bin/julia
using JSON
using LinearAlgebra
using MatrixMarket
using SparseArrays
using Random
using Printf

# The difference to `generate_random_partition` in python/partition.py is that
# all samples have the same amount of partitions (no merging of partition boundaries)
function generate_holes(rng, n_last, n_holes, n_min_part, n_max_part, n_samples)
    holes = Vector{Vector{Int64}}()
    n_holes_done = 0
    attempts = 0

    while n_holes_done < n_samples
        # TODO: find more efficient way to sample
        sample = sort(vcat([1; randperm(N)[1:n_holes]; n_last])) # sorted array of length n_holes+2
        is_valid = true  # determines if a partition is of a given size

        for (part_first, part_last) in Iterators.zip(sample, sample[2:length(sample)])
            part_size = part_last - part_first + 1
    
            if !(part_size <= n_max_part && part_size >= n_min_part)
                is_valid = false
                break
            end
        end

        if is_valid == true
            # copy because we reassign `sample` in every iteration
            push!(holes, convert(Vector{Int64}, sample[2:n_holes+1]))
            println("sample [" * string(n_holes) * "] generated of size " * string(n_min_part) * " <= M <= " * string(n_max_part) * " [" * string(n_holes_done) * "]")
            n_holes_done += 1
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
# TODO: extend rhs with more entries if possible
N = 512
idx = [11, 14]
rng = MersenneTwister(1234)
rhs = [ones(N), zeros(N), randn(rng, N)]

# generate "holes" in matrix
#n_holes_min = 8  # corresponding to partition size of M=64
#n_holes_max = 16 # corresponding to partition size of M=32
n_holes = 16
n_part_min_size = 16
n_part_max_size = 80
n_holes_samples = 1000 # number of samples within partition bounds

# holes are generated independently from linear systems
# rng = MersenneTwister(1234)
# holes = generate_holes(rng, N, n_holes, n_part_min_size, n_part_max_size, n_holes_samples; 
#                        n_max_attempts=50000000)
# @assert sum(unique(map(length, holes))) == n_holes
# @assert length(unique(holes)) == length(holes)

n_holes_range = 8:16
v_holes = Vector{Vector{Vector{Int64}}}(undef, length(n_holes_range))

# Generate samples for varying amount of holes
# Threads.@threads for n_holes in n_holes_range
#     rng = MersenneTwister(1234)
#     holes = generate_holes(rng, N, n_holes, n_part_min_size, n_part_max_size, n_holes_samples)

#     @assert sum(unique(map(length, holes))) == n_holes
#     @assert length(unique(holes)) == length(holes)
#     v_holes[n_holes-n_holes_range[1]+1] = holes  # n_holes partitioned between threads

#     jname = @sprintf("holes-%i-%i-%i-%02i.json", N, n_part_min_size, n_part_max_size, n_holes)
#     open(jname, "w") do f
#         JSON.print(f, holes)
#     end
# end

for n_holes in n_holes_range
    jname = @sprintf("decoupled/holes/holes-%i-%i-%i-%02i.json", N, n_part_min_size, n_part_max_size, n_holes)
    holes = JSON.parsefile(jname)

    @assert sum(unique(map(length, holes))) == n_holes
    @assert length(unique(holes)) == length(holes)
    v_holes[n_holes-n_holes_range[1]+1] = holes  # n_holes partitioned between threads
end

# set rows determined by hole indices to 0 (later: small epsilon)
for mtx_id in idx
    for n_holes in n_holes_range
        S = MatrixMarket.mmread("mtx/" * string(mtx_id) * "-" * string(N) * ".mtx")
        S_dl, S_d, S_du = diag(S, -1), diag(S), diag(S, 1)
        holes = v_holes[n_holes-n_holes_range[1]+1]

        Threads.@threads for k in 1:n_holes_samples
            sample = holes[k]
            dl, d, du = copy(S_dl), copy(S_d), copy(S_du)
            dl[sample] .= 0.0
            du[sample] .= 0.0

            # push!(S_samples, dropzeros(SparseMatrixCSC(Tridiagonal(dl, d, du))))
            fname = @sprintf("mtx-%i-%i-decoupled-%i-%i-%02i-%04i.mtx", mtx_id, N, n_part_min_size, n_part_max_size, n_holes, k)
            S_new = dropzeros(SparseMatrixCSC(Tridiagonal(dl, d, du)))
            S_new_cond = cond(Array(S_new), 2)
            MatrixMarket.mmwrite(fname, S_new)

            # TODO: if the rhs is zero, it suffices to set zero as the solution vector
            for (bi, b) in enumerate(rhs)
                jname = @sprintf("mtx-%i-%i-decoupled-%i-%i-%02i-%04i-rhs%i.json", mtx_id, N, n_part_min_size, n_part_max_size, n_holes, k, bi) # 1-indexed positions
                sol, res, acc = tridiag_exact_solution(S_new, b)

                open(jname, "w") do f
                    JSON.print(f, Dict("sample_1idx" => sample, "solution"  => sol, "max_accuracy" => acc, 
                                        "residual"   => res,    "condition" => S_new_cond, 
                                        "rhs"        => b,      "n_holes"   => n_holes))
                end
            end
        end
    end
end
