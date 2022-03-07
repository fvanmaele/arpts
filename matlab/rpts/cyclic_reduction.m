function x = cyclic_reduction(a, b, c, d, varargin)
    pivoting = 'scaled_partial';
    if length(varargin) == 1
        pivoting = varargin{1};
    end
    
    switch pivoting
        case 'scaled_partial'
            
        case 'partial'
            mu = 1;
            mm = 1;
            ml = 1;

        case 'none'
            mu = 0;
            mm = 0;
            ml = 0;

        otherwise
            error("unknown pivoting method")
    end
end