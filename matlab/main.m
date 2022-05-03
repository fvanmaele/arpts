%% Write matrices to Matrix Market (MM) files
rng('default')  % default seed (0) and algorithm (Mersenne Twister)

for N = [512 2048]
    for Id = 1:20
        fprintf("Generating matrix %02d, N = %d\n", Id, N)
        fname = sprintf("%02d-%d.mtx", Id, N);

        A = generate_matrix(Id, N);
        mmwrite(fname, A)
    end
end