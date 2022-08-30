randn('state', 0);
rand('state', 0);
n = 500;
% generate a well-conditioned positive definite matrix
% (for faster convergence)
P = rand(n);
P = P + P';
[V D] = eig(P);
P = V*diag(1+rand(n,1))*V';

q = randn(n,1);
r = randn(1);

l = randn(n,1);
u = randn(n,1);
lb = min(l,u);
ub = max(l,u);

[x history] = quadprog(P, q, r, lb, ub, 1.0);       % the original ADMM
[x history] = quadprog_pd_pc(P, q, r, lb, ub, 1.0); % the primal-dual extended ADMM
[x history] = quadprog_dp_pc(P, q, r, lb, ub, 1.0); % the dual-primal extended ADMM