randn('state', 0);
rand('state', 0);

n = 500;  % dimension of x
m = 300;  % number of equality constraints

c  = rand(n,1) + 0.5;    % create nonnegative price vector with mean 1
x0 = abs(randn(n,1));    % create random solution vector

A = abs(randn(m,n));     % create random, nonnegative matrix A
b = A*x0;
[x history] = linprog(c, A, b, 1.0);       % the original ADMM
[x history] = linprog_pd_pc(c, A, b, 1.0); % the primal-dual extended ADMM
[x history] = linprog_dp_pc(c, A, b, 1.0); % the dual-primal extended ADMM