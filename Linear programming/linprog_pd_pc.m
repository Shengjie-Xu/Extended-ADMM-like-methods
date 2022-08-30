function [z, history] = linprog_pd_pc(c, A, b, rho)
% Solves the following problem via the primal-dual extended ADMM:
%
%   minimize     c'*x
%   subject to   Ax = b, x >= 0
%
nu=0.99;
t_start = tic;
QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
[m n] = size(A);

x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    % prediction step
    % x-update
    tmp = [ rho*eye(n), A'; A, zeros(m) ] \ [ rho*x + u - c; b ];
    xx = tmp(1:n);
    % z-update 
    zold = z;
    zz = pos(xx-x+zold - u/rho);
    % u-update
    uu = u - rho*(xx - zz);
    u=uu+nu*rho*(x-xx);
    % correction step
    x=x-nu*(x-xx+z-zz);
    z=z+nu*(zz-z);
    
    % diagnostics, reporting, termination checks

    history.objval(k)  = objective(c, x);
    history.r_norm(k)  = norm(x - z);
    history.s_norm(k)  = norm(-rho*(z - zold));

    history.eps_pri(k) = sqrt(n)*ABSTOL + RELTOL*max(norm(x), norm(-z));
    history.eps_dual(k)= sqrt(n)*ABSTOL + RELTOL*norm(rho*u);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end

    if (history.r_norm(k) < history.eps_pri(k) && ...
       history.s_norm(k) < history.eps_dual(k))
         break;
    end
end

if ~QUIET
    toc(t_start);
end
end

function obj = objective(c, x)
    obj = c'*x;
end

