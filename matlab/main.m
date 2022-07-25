%% Write matrices to Matrix Market (MM) files
rng('default')  % default seed (0) and algorithm (Mersenne Twister)

for N = [512 2048]
    for Id = 1:20
        gen_matrix(Id, N)
    end
end

%% Additional cases
for N = [512 2048]
    for Id = 21:30
        gen_matrix(Id, N);
    end
end

%%
function gen_matrix(Id, N)
    fprintf("Generating matrix %02d, N = %d\n", Id, N)
    fname = sprintf("../mtx/%02d-%d.mtx", Id, N);
    S = generate_matrix(Id, N);
    mmwrite(fname, S)
end

