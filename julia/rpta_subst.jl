#!/usr/bin/julia

# x1_prev_partition = 0 for the first partition
# x0_next_partition = 0 for the last partition
function eliminate_band_with_solution(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector,
        x1_prev_partition, x0, x1, x0_next_partition, pivoting)
    M = length(a)
    @assert M > 1 "band should have at least one element"

    x = zeros(Float64, M)
    x[M] = x1
    x[1] = x0

    # Substitute with solution
    d[M-1] = d[M-1] - c[M-1]*x[M]
    c[M-1] = 0

    s_c = zeros(Float64, 5)
    s_p = zeros(Float64, 5)

    s_p[1] = 0.0
    s_p[2] = b[2]
    s_p[3] = c[2]
    s_p[4] = 0.0
    s_p[5] = d[2] = a[2]*x0
    ip = 2
    i = fill(nothing, M) # XXX: mixed types

    # Downwards oriented elimination
    for j = 3:1:M-1
        s_c[1] = 0.0
        s_c[2] = a[j]
        s_c[3] = b[j]
        s_c[4] = c[j]
        s_c[5] = d[j]

        if pivoting == "scaled_partial"
            mp = max([abs(s_p[2]), abs(s_p[3])])
            mc = max([abs(s_c[2]), abs(s_c[3]), abs(s_c[4])])
        elseif pivoting == "partial"
            mp = 1.0
            mc = 1.0
        elseif pivoting == "none"
            mp = 0.0
            mc = 0.0
        else
            throw(ArgumentError("unknown pivoting method"))
        end

        if abs(s_c[2])*mp <= abs(s_p[2])*mc
            i[j - 1] = ip
            r_c = 1.0
            r_p = -s_c[2] / s_p[2]
            
            a[ip] = s_p[2]
            b[ip] = s_p[3]
            c[ip] = 0.0
            d[ip] = s_p[5]
            ip = j
        else
            i[j - 1] = j
            r_c = -s_p[2] / s_c[2]
            r_p = 1
        end

        for k in [3, 4, 5]
            s_p[k] = r_p*s_p[k] + r_c*s_c[k]
        end
        s_p[2] = s_p[3]
        s_p[3] = s_p[4]
        s_p[4] = 0
    end

    # Pivoting
    if abs(s_p[2]) >= abs(a[M])
        x[M-1] = s_p(5) / s_p[2]
    else
        x[M-1] = (d[M] - b[M]*x[M] - c[M]*x0_next_partition) / a[M]
    end

    # Upwards oriented substitution
    for j in M-2:-1:2
        k = i[j]
        x[j] = (d[k] - b[k]*x[j + 1] - c[k]*x[j + 2]) / a[k]
    end
    
    # Pivoting
    # XXX: hack taken from the Python code
    k = i[2]
    if !isnothing(k) && abs(a(k)) >= abs(c[1])
        x[2] = (d[k] - b[k]*x[3] - c[k]*x[4]) / a[k]
    elseif !isnothing(k)
        x[2] = (d[1] - b[1]*x[1] - a[1]*x1_prev_partition) / c[1]
    end
end

# note: in julia array intervals are closed, not half-open
# v = [1, 2, 3], v[1:2] => [1, 2]
function rpta_substitute(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector,
        x_coarse::AbstractVector; pivoting="scaled_partial")
        num_partitions = len(partition)
    N = len(a_fine)
    x_fine = [None] * N

    for (partition_id, partition_bounds) in enumerate(partition)
        partition_begin = partition_bounds[1]
        partition_last = partition_bounds[2]

        if partition_id > 1
            x1_prev_partition = x_coarse[partition_id * 2 - 2]
        else
            x1_prev_partition = 0.0
        end

        if partition_id < num_partitions - 1
            x0_next_partition = x_coarse[partition_id * 2 + 1]
        else
            x0_next_partition = 0.0
        end

        x0 = x_coarse[partition_id * 2 - 1]
        x1 = x_coarse[partition_id * 2]

        x_partition = eliminate_band_with_solution(
            list(a_fine[partition_begin:partition_last]),
            list(b_fine[partition_begin:partition_last]),
            list(c_fine[partition_begin:partition_last]),
            list(d_fine[partition_begin:partition_last]),
            x1_prev_partition, x0, x1, x0_next_partition, pivoting)

        x_fine[partition_begin:partition_last] = x_partition
    end

    return x_fine
end