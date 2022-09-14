%% kernel_substitution
% x1_prev_partition = 0 for the first partition
% x0_next_partition = 0 for the last partition
function [x] = kernel_substitution(a, b, c, d, x1_prev_partition, x0, x1, ...
                                   x0_next_partition, varargin)
    % Precondition checks
    if ~isvector(a) || ~isvector(b) || ~isvector(c) || ~isvector(d)
        error("a, b, c, d must be vectors")
    end

    if ~isscalar(x1_prev_partition) || ~isscalar(x0) || ~isscalar(x1) || ...
       ~isscalar(x0_next_partition)
        error("x0, x1, x0_next_partition, x1_prev_partition must be scalars")
    end

    M = length(a);
    if ~length(b) == M || ~length(c) == M || ~length(d) == M
        error("a, b, c, d must be of length " + M)
    end

    % Set optional parameters
    n_varargs = length(varargin);
    threshold = 0;
    pivoting = 'scaled_partial';
    
    if n_varargs == 1
        threshold = varargin{1};
    elseif n_varargs == 2
        threshold = varargin{1};
        pivoting = varargin{2};
    end

    % Initialize solution vector
    x = zeros(M, 1);
    x(M) = x1;
    x(1) = x0;

    % Substitute with solution
    d(M-1) = d(M-1) - c(M-1)*x(M);
    c(M-1) = 0;
    sp = [0 b(2) c(2) 0 (d(2) - a(2)*x(1))];
    ip = 2;

    for j = 3:M-1
        sc = [0 a(j) b(j) c(j) d(j)];
        apply_threshold(threshold, sp(2), sc(2));

        switch pivoting
            case 'scaled_partial'
                mp = max([abs(sp(2)), abs(sp(3))]);
                mc = max([abs(sc(2)), abs(sc(3)), abs(sc(4))]);
            case 'partial'
                mp = 1;
                mc = 1;
            case 'none'
                mp = 0;
                mc = 0;
            otherwise
                error("unknown pivoting method")
        end

        if abs(sc(2))*mp <= abs(sp(2))*mc
            a(ip) = sp(2);
            b(ip) = sp(3);
            c(ip) = 0;
            d(ip) = sp(5);
            i(j-1) = ip;
            rp = -sc(2)/sp(2);
            rc = 1;
            ip = j;
        else
            i(j-1) = j;
            rp = 1;
            rc = -sp(2)/sc(2);
        end

        for k = 3:5
            sp(k) = rp*sp(k) + rc*sc(k);
        end
        sp(2) = sp(3);
        sp(3) = sp(4);
        sp(4) = 0;
    end

    % Pivoting
    if abs(sp(2)) >= abs(a(M))
        x(M-1) = sp(5)/sp(2);
    else
        x(M-1) = (d(M)-b(M)*x(M)-c(M)*x0_next_partition) / a(M);
    end

    for j = M-2:-1:2
        k = i(j);
        x(j) = (d(k) - b(k)*x(j+1) - c(k)*x(j+2)) / a(k);
    end
    
    % Pivoting
    k = i(2);
    if abs(a(k)) >= abs(c(1))
        x(2) = (d(k) - b(k)*x(3) - c(k)*x(4)) / a(k);
    else
        x(2) = (d(1) - b(1)*x(1) - a(1)*x1_prev_partition) / c(1);
    end
end