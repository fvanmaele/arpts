%% array_to_bands
% Convert a 2D MATLAB array representing a tridiagonal matrix to three
% arrays containing the lower, middle and upper band. The three arrays
% are padded with zeros to have equal length.
%%
% Representation for a matrix $A\in R^{N\times N}$
%
%     / b[0]    c[0]                           \
%     | a[1]    b[1]    c[1]                   |
%     |         a[2]    b[2]    c[2]           |
% A = |         ...     ...     ...            |
%     |                 a[N-2]  b[N-2]  c[N-2] |
%     \                         a[N-1]  b[N-1] /
%
function [a, b, c] = array_to_bands(A)
    % Precondition checks
    if ~isbanded(A, 1, 1)
        error("A must be a tridiagonal matrix")
    end

    a = [0; diag(A, -1)];
    b = diag(A, 0);
    c = [diag(A, 1); 0];
end