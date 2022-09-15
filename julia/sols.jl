#!/usr/bin/julia
using JSON
using LinearAlgebra
using SparseArrays
using MatrixMarket
using Random

# load linear system
N = 512

# generate random right-hand sides
rng = MersenneTwister(1234)
rhs = Vector{Vector{Float64}}()
for i in 1:1000
    if i % 50 == 0
        println("rhs [", i, "]")
    end
    push!(rhs, randn(rng, Float64, N))
end

# save to disk for later processing
open("00_rhs.json", "w") do f
    JSON.print(f, Dict("rhs" => rhs))
end

# all further calculations will take precision changes into account
setprecision(512)

for name in ARGS
    M = MatrixMarket.mmread("mtx/" * name * ".mtx")
    S = Tridiagonal(diag(M, -1), diag(M), diag(M, 1))
    
    # extended precision matrix inverse
    Su = convert(Tridiagonal{BigFloat}, S)
    Su_inv = inv(Su)
    
    # compute corresponding solutions
    sol = Vector{Vector{BigFloat}}()
    for i in 1:1000
        if i % 50 == 0
            println("Id: ", name, ", sol [", i, "]")
        end
        push!(sol, Su_inv * rhs[i])
    end
    
    # sanity check: upper bound on forward relative error
    maxacc = BigFloat(0)
    for i in 1:1000
        if i % 50 == 0
            println("Id: ", name, ", acc [", i, "]")
        end
        res = norm(Su * sol[i] - rhs[i])  # residual
        acc = norm(Su_inv) * (res / (norm(sol[i])))
        if acc > maxacc
            maxacc = acc
        end
    end
    println("Id: ", name, ", maxacc: ", maxacc)

    open(name * ".json", "w") do f
        JSON.print(f, Dict("solutions" => sol, "maxacc" => maxacc))
    end
end
