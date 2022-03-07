%% apply_threshold
% Map coefficients smaller than eps to zero. This option allows the user
% to increase numeric stability in the case of noisy input coefficients.
% Setting eps = 0 switches off this behavior.
%
%% Input
% eps:          threshold
% x:            scalar, mapped to smallest positive normalized (floating-point) value
% y:            scalar, mapped to zero
% precision:    'single' or 'double' (default: 'double)
%
function [x, y] = apply_threshold(eps, x, y, varargin)
    if eps == 0
        return
    end
    n_varargs = length(varargin);
    precision = 'double';

    if n_varargs == 1
        precision = varargin{1};
    end
    if abs(x) <= eps
        x = realmin(precision);
    end    
    if abs(y) <= eps
        y = 0;
    end
end
