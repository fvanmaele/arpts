%% generate_matrix
function A = generate_matrix(ID, N)
    b_unif = unifrnd(-1, 1, N, 1);
    a_unif = unifrnd(-1, 1, N-1, 1);
    c_unif = unifrnd(-1, 1, N-1, 1);
    
    switch ID
        case '1'
            A = tridiag(a_unif, b_unif, c_unif);
        
        case '2'
            b = 1e8*ones(N, 1);
            A = tridiag(a_unif, b, c_unif);
        
        case '3'
            A = gallery('lesp', N);
        
        case '4'
            a = a_unif;
            % XXX: use ceil() for 1-indexing?floor
            a(floor(N/2)) = 1e-50*a_unif(floor(N/2));
            a(floor(N/2)+1) = 1e-50*a_unif(floor(N/2)+1);

            A = tridiag(a, b_unif, c_unif);

        case '5'
            % each element of a has 50% chance to be zero
            mask_a = logical(binornd(1, 0.5, N-1, 1));
            a = zeros(N-1, 1);
            a(mask_a) = a_unif(mask_a);

            % each element of c has 50% chance to be zero
            mask_c = logical(binornd(1, 0.5, N-1, 1));
            c = zeros(N-1, 1);
            c(mask_c) = c_unif(mask_c);

            A = tridiag(a, b_unif, c);

        case '6'
            b = 64*ones(N, 1);
            A = tridiag(a_unif, b, c_unif);

        case '7'
            A = inv(gallery('kms',  N, 0.5));

        case '8'
            A = gallery('randsvd', N, 1e15, 2, 1, 1);

        case '9'
            A = gallery('randsvd', N, 1e15, 3, 1, 1);

        case '10'
            A = gallery('randsvd', N, 1e15, 1, 1, 1);

        case '11'
            A = gallery('randsvd', N, 1e15, 4, 1, 1);

        case '12'
            a = a_unif*1e-50;
            A = tridiag(a, b_unif, c_unif);

        case '13'
            A = gallery('dorr', N, 1e-4);

        case '14'
            b = 1e-8*ones(N, 1);
            A = tridiag(a_unif, b, c_unif);

        case '15'
            b = zeros(N, 1);
            A = tridiag(a_unif, b, c_unif);
        
        case '16'
            A = tridiag(ones(N-1, 1), 1e-8*ones(N, 1), ones(N-1, 1));

        case '17'
            A = tridiag(ones(N-1, 1), 1e8*ones(N, 1), ones(N-1, 1));

        case '18'
            A = tridiag(-ones(N-1, 1), 4*ones(N, 1), -ones(N-1, 1));

        case '19'
            A = tridiag(-ones(N-1, 1), 4*ones(N, 1), ones(N-1, 1));

        case '20'
            A = tridiag(-ones(N-1, 1), 4*ones(N, 1), c_unif);
    end
end

%% tridiag
function A = tridiag(a, b, c, varargin)
    % Default to dense matrix
    sparse_result = 0;
    if length(varargin) == 1
        sparse_result = varargin{1};
    end

    % Precondition checks
    N = length(b);    
    if ~isvector(a) || ~isvector(b) || ~isvector(c)
        error("a, b, c must be vectors")
    end
    
    if length(a) ~= N-1
        error("a must have |a|-1 elements")
    end
    
    if length(c) ~= N-1
        error("c must have |a|-1 elements")
    end
    
    if sparse_result == 1
        B = [[a; 0] b [0; c]];
        A = spdiags(B, -1:1, N, N);
    else
        A = diag(a, -1) + diag(b) + diag(c, 1);
    end
end