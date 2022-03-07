%% kernel_reduction
% Reduction phase for one partition
%
%% Input
% a:        lower band of partition
% b:        middle band of partition
% c:        upper band of partition 
% d:        right-hand side
%
%% Output
% lower:    lower row of coarse system for partition
% upper:    upper row of coarse system for partition
% rhs:      right-hand-side
%
function [a_coarse, b_coarse, c_coarse, d_coarse] = kernel_reduction(a_fine, b_fine, c_fine, d_fine, M, varargin)
    N = length(a_fine);
    N_coarse = 2*N/M + mod(N, M);

    a_coarse = zeros(N_coarse, 1);
    b_coarse = zeros(N_coarse, 1);
    c_coarse = zeros(N_coarse, 1);
    d_coarse = zeros(N_coarse, 1);

    partition_range = length(1:M:N);
    for partition_id = 1:length(partition_range)
        partition_offset = partition_range(partition_id);
        partition_end = min(partition_offset+M, N);
        
        [a_coarse_lower, b_coarse_lower, c_coarse_lower, d_coarse_lower] = ...
            eliminate_band(a_fine(partition_offset:partition_end), ...
                           b_fine(partition_offset:partition_end), ...
                           c_fine(partition_offset:partition_end), ...
                           d_fine(partition_offset:partition_end));

        [c_coarse_upper, b_coarse_upper, a_coarse_upper, d_coarse_upper] = ...
            eliminate_band(flip(c_fine(partition_offset:partition_end)), ...
                           flip(b_fine(partition_offset:partition_end)), ...
                           flip(a_fine(partition_offset:partition_end)), ...
                           flip(d_fine(partition_offset:partition_end)));

        a_coarse(2 * partition_id) = a_coarse_upper;
        b_coarse(2 * partition_id) = b_coarse_upper;
        c_coarse(2 * partition_id) = c_coarse_upper;
        d_coarse(2 * partition_id) = d_coarse_upper;

        a_coarse(2 * partition_id + 1) = a_coarse_lower;
        b_coarse(2 * partition_id + 1) = b_coarse_lower;
        c_coarse(2 * partition_id + 1) = c_coarse_lower;
        d_coarse(2 * partition_id + 1) = d_coarse_lower;
    end
end