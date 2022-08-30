function [z, history] = quadprog_dp_pc(P, q, r, lb, ub, rho)

% Solves the following problem via dual-primal extended ADMM:
%
%   minimize     (1/2)*x'*P*x + q'*x + r
%   subject to   lb <= x <= ub
%
nu=0.99;
t_start = tic;
QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
n = size(P,1);
x = zeros(n,1);
z = zeros(n,1);
u = zeros(n,1);

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    
    % prediction step
    % u-update
    uu = u - rho*(x - z);
    % x-update
    if k > 1
        xx = R \ (R' \ (rho*x + uu - q));
    else
        R = chol(P + rho*eye(n));
        xx = R \ (R' \ (rho*x + uu - q));
    end
    % z-update 
    zold = z;
    zz = min(ub, max(lb, xx-x+zold - uu/rho));
    
    % correction step
    u=uu+rho*(x-xx+zz-z);
    x=x-nu*(x-xx+z-zz);
    z=z+nu*(zz-z);
    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(P, q, r, x);

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

function obj = objective(P, q, r, x)
    obj = 0.5*x'*P*x + q'*x + r;
end
