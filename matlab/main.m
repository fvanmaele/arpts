%% Write matrices to Matrix Market (MM) files
N = 2048;
rng('default')  % default seed (0) and algorithm (Mersenne Twister)

for Id = 1:20
    fprintf("Generating matrix %02d\n", Id)
    fname = sprintf("%02d-%d.mtx", Id, N);
    A = generate_matrix(Id, N);
    mmwrite(fname, A)
end
