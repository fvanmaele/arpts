#!/usr/bin/julia
using JSON
using LinearAlgebra
using MatrixMarket
using SparseArrays
using Random
using Printf
using ArgParse

function parse_commandline
    s = ArgParseSettings(autofix_names=true)

    @add_arg_table! s begin
        "N"
            help = "matrix dimension"
            arg_type = Int
            required = true
        "id"
            help = "matrix ID number"
            arg_type = Int
            required = true
        "n-holes"
            help = "number of holes (decoupled systems) in each sample"
            arg_type = Int
            required = true
        "eps"
            help = "factor for multiplying elements at partition boundaries"
            arg_type = Float64
            required = true
        "--min-part"
            help = "minimum size of partition blocks"
            arg_type = Int
            default = 16
        "--max-part"
            help = "maximum size of partition blocks"
            arg_type = Int
            default = 80
        "--n-samples"
            help = "number of generated matrix samples"
            arg_type = Int
            default = 1000
        "--seed"
            help = "seed for random number generator"
            arg_type = Int
            default = 1234
    end

    return parse_args(s)
end

# The difference to `generate_random_partition` in python/partition.py is that
# all samples have the same amount of partitions (no merging of partition boundaries)
function generate_holes(rng, n_last, n_holes, n_part_min, n_part_max, n_samples)
    holes = Vector{Vector{Int64}}()
    n_holes_done = 0
    attempts = 0

    while n_holes_done < n_samples
        sample = sort(vcat([1; randperm(N)[1:n_holes]; n_last])) # sorted array of length n_holes+2
        is_valid = true  # determines if a partition is of a given size

        for (part_first, part_last) in Iterators.zip(sample, sample[2:length(sample)])
            part_size = part_last - part_first + 1
    
            if !(part_size <= n_part_max && part_size >= n_part_min)
                is_valid = false
                break
            end
        end

        if is_valid == true
            # copy because we reassign `sample` in every iteration
            push!(holes, convert(Vector{Int64}, sample[2:n_holes+1]))
            println("sample [" * string(n_holes) * "] generated of size " * string(n_part_min) * " <= M <= " * string(n_part_max) * " [" * string(n_holes_done) * "]")
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

# TODO: use main function with command-line arguments
function main
    parsed_args = parse_commandline()
    
    # linear system with fixed right-hand side
    N    = parsed_args["N"]
    idx  = parsed_args["id"]
    seed = parsed_args["seed"]
    eps  = parsed_args["eps"]
    rng  = MersenneTwister(seed)
    rhs  = [ones(N), randn(rng, N), map(sinpi, LinRange(0, 200, N))]
    
    # generate "holes" in matrix
    n_part_min_size = parsed_args["n_part_min"]
    n_part_max_size = parsed_args["n_part_max"]
    n_samples = parsed_args["n_samples"]
    n_holes = parsed_args["n_holes"]
    v_holes = Vector{Vector{Vector{Int64}}}(undef, length(n_holes_range))

    # Generate samples for given amount of holes
    rng = MersenneTwister(seed)
    jname = @sprintf("decoupled/holes/holes-%i-%i-%i-%02i.json", N, n_part_min_size, n_part_max_size, n_holes)

    if isfile(jname)
        holes = JSON.parsefile(jname)
    else
        holes = generate_holes(rng, N, n_holes, n_part_min_size, n_part_max_size, n_samples)
        @assert sum(unique(map(length, holes))) == n_holes
        @assert length(unique(holes)) == length(holes)

        open(jname, "w") do f
            JSON.print(f, holes)
        end
    end

    # eps_range = [0, 1e-16, 1e-12, 1e-8, 1e-4]
    S = MatrixMarket.mmread("mtx/" * string(mtx_id) * "-" * string(N) * ".mtx")
    S_dl, S_d, S_du = diag(S, -1), diag(S), diag(S, 1)

    Threads.@threads for k in 1:n_samples  # controlled by julia --threads=<N>
        sample = holes[k]
        dl, d, du = copy(S_dl), copy(S_d), copy(S_du)
        dl[sample] .= eps * dl[sample]
        dl[sample] .= eps * dl[sample]

        # push!(S_samples, dropzeros(SparseMatrixCSC(Tridiagonal(dl, d, du))))
        fname = @sprintf("mtx-%i-%i-decoupled-%i-%i-%02i-%2.2e-%04i.mtx", 
                          mtx_id, N, n_part_min_size, n_part_max_size, n_holes, eps, k)
        S_new = dropzeros(SparseMatrixCSC(Tridiagonal(dl, d, du)))
        S_new_cond = cond(Array(S_new), 2)
        MatrixMarket.mmwrite(fname, S_new)

        for (bi, b) in enumerate(rhs)
            jname = @sprintf("mtx-%i-%i-decoupled-%i-%i-%02i-%2.2e-%04i-rhs%i.json", 
                              mtx_id, N, n_part_min_size, n_part_max_size, n_holes, eps, k, bi) # 1-indexed positions
            sol, res, acc = tridiag_exact_solution(S_new, b)

            open(jname, "w") do f
                JSON.print(f, Dict("sample_1idx" => sample, "solution"  => sol, "max_accuracy" => acc, 
                                    "residual" => res, "condition" => S_new_cond,  "rhs" => b, "n_holes" => n_holes, 
                                    "eps" => eps, "N" => N, "mtx_id" => mtx_id, "seed" => seed,
                                    "n_part_min" => n_part_min_size, "n_part_max" => n_part_max_size))
            end
        end
    end
end
