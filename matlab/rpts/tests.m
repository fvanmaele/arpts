clear; clc

%% Basic test (single recursion level, N = 21, M = 7)
N = 21;
n = N^2;
M = 7;
A = diag(N:2*N-1) + diag(ones(N-1,1),1) + diag(ones(N-1,1),-1);
[a, b, c] = array_to_bands(A);
r = normrnd(0, 1, N, 1); % right-hand side

%% Classic solution (QR decomposition)
x = A \ r;

%% Condition of each partition
A1 = A(1:7, 1:7);
A2 = A(8:14, 8:14);
A3 = A(15:21, 15:21);
A1_cond = cond(A1);
A2_cond = cond(A2);
A3_cond = cond(A3);

%% Partition arrays (M = 7)
[l1, u1, r1] = kernel_reduction(a(1:7), b(1:7), c(1:7), r(1:7));
[l2, u2, r2] = kernel_reduction(a(8:14), b(8:14), c(8:14), r(8:14));
[l3, u3, r3] = kernel_reduction(a(15:21), b(15:21), c(15:21), r(15:21));

%% Partition arrays (M = 10)
[l11, u11, r11] = kernel_reduction(a(1:10), b(1:10), c(1:10), r(1:10));
[l21, u21, r21] = kernel_reduction(a(11:21), b(11:21), c(11:21), r(11:21));

%% Coarse system (M = 7)
r_grob = [r1 r2 r3]';
A_grob = [u1; l1; u2; l2; u3; l3];
x_grob = A_grob \ r_grob;

%% Coarse system (M = 10)
r1_grob = [r11 r21]';
A1_grob = [u11; l11; u21; l21];
x1_grob = A1_grob \ r1_grob;
