%% eliminate_band
%
%% Input
% a:    lower band of partition
% b:    middle band of partition
% c:    upper band of partition 
% d:    right-hand side
% Optional arguments:
%
%% Output
%
function [s1, s2, s3, s4] = eliminate_band(a, b, c, d, varargin)
    % Precondition checks
    if ~isvector(a) || ~isvector(b) || ~isvector(c) || ~isvector(d)
        error("a, b, c, d must be vectors")
    end

    M = length(a);
    if ~length(b) == M || ~length(c) == M || ~length(d) == M
        error("a, b, c, d must be of length " + M)
    end

    % Temporary register arrays for c(urrent) and p(revious) values
    sp = [a(2) b(2) c(2) 0 d(2)];
    
    % Set optional parameters (threshold, pivoting type)
    varargin = varargin{1}; % ???
    n_varargs = length(varargin);
    threshold = 0;
    pivoting = 'scaled_partial';
    
    if n_varargs == 1
        threshold = varargin{1};
    elseif n_varargs == 2
        threshold = varargin{1};
        pivoting = varargin{2};
    end

    M = length(a);
    for j = 3:M
        sc = [0 a(j) b(j) c(j) d(j)];
        apply_threshold(threshold, sp(2), sc(2));
    
        switch pivoting
            case 'scaled_partial'
                mp = max([abs(sp(1)), abs(sp(2)), abs(sp(3))]);
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
            rp = -sc(2) / sp(2);
            rc = 1;
        else
            rp = 1;
            rc = -sp(2) / sc(2);
        end

        for k = 3:5
            sp(k) = rp*sp(k) + rc*sc(k);
        end
        
        sp(1) = rp*sp(1);
        sp(2) = sp(3);
        sp(3) = sp(4);
        sp(4) = 0;
    end
    
    s1 = sp(1);
    s2 = sp(2);
    s3 = sp(3);
    s4 = sp(5);
end