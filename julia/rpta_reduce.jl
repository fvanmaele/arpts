#!/usr/bin/julia

function eliminate_band(a::AbstractVector, b::AbstractVector, c::AbstractVector, d::AbstractVector, pivoting::AbstractString)
    # Temporary register arrays for c(urrent) and p(revious) values
    sp = [a[2] b[2] c[2] 0 d[2]]
    
    M = length(a)
    @assert M > 1 "band should have at least one element"

    for j = 3:1:M
        sc = [0 a[j] b[j] c[j] d[j]]
    
        if pivoting == "scaled_partial"
            mp = max([abs(sp[1]), abs(sp[2]), abs(sp[3])])
            mc = max([abs(sc[2]), abs(sc[3]), abs(sc[4])])
        elseif pivoting == "partial"
            mp = 1.0
            mc = 1.0
        elseif pivoting == "none"
            mp = 0.0
            mc = 0.0
        else
            throw(ArgumentError("unknown pivoting method"))
        end

        if abs(sc[2])*mp <= abs(sp[2])*mc
            r_p = -sc[2] / sp[2]
            r_c = 1.0
        else
            r_p = 1.0
            r_c = -sp[2] / sc[2]
        end

        for k in [3, 4, 5]
            sp[k] = r_p*sp[k] + r_c*sc[k]
        end
        
        sp[1] = r_p*sp[1]
        sp[2] = sp[3]
        sp[3] = sp[4]
        sp[4] = 0.0
    end
    
    s1 = sp[1]
    s2 = sp[2]
    s3 = sp[3]
    s4 = sp(5)
end

# note: in julia array intervals are closed, not half-open
# v = [1, 2, 3], v[1:2] => [1, 2]
function r_pta_reduce(a_fine::AbstractVector, b_fine::AbstractVector, c::AbstractVector, d::AbstractVector,
        a_coarse::AbstractVector, b_coarse::AbstractVector, c_coarse::AbstractVector,
        partition::AbstractVector; pivoting="scaled_partial")
    for (partition_id, partition_bounds) in enumerate(partition)
        partition_begin = partition_bounds[1]
        partition_last = partition_bounds[2]

        a_coarse_lower, b_coarse_lower, c_coarse_lower, d_coarse_lower = eliminate_band(
            a_fine[partition_begin:partition_last],
            b_fine[partition_begin:partition_last],
            c_fine[partition_begin:partition_last],
            d_fine[partition_begin:partition_last],
            pivoting)

        # reverse: copy of v reversed from start to stop
        c_coarse_upper, b_coarse_upper, a_coarse_upper, d_coarse_upper = eliminate_band(
            reverse(c_fine[partition_begin:partition_last]),
            reverse(b_fine[partition_begin:partition_last]),
            reverse(a_fine[partition_begin:partition_last]),
            reverse(d_fine[partition_begin:partition_last]),
            pivoting)

        a_coarse[2 * partition_id - 1] = a_coarse_upper
        b_coarse[2 * partition_id - 1] = b_coarse_upper
        c_coarse[2 * partition_id - 1] = c_coarse_upper
        d_coarse[2 * partition_id - 1] = d_coarse_upper

        a_coarse[2 * partition_id] = a_coarse_lower
        b_coarse[2 * partition_id] = b_coarse_lower
        c_coarse[2 * partition_id] = c_coarse_lower
        d_coarse[2 * partition_id] = d_coarse_lower
    end
end