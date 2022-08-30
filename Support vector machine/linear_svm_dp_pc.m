function [xave, history] = linear_svm_dp_pc(A, lambda, p, rho)

% the dual-primal extended ADMM
% More information can be found in the paper linked at:
% http://www.stanford.edu/~boyd/papers/distr_opt_stat_learning_admm.html
%
nu=0.99;
t_start = tic;
QUIET    = 0;
MAX_ITER = 1000;
ABSTOL   = 1e-4;
RELTOL   = 1e-2;
[m, n] = size(A);
N = max(p);
% group samples together
for i = 1:N
    tmp{i} = A(p==i,:);
end
A = tmp;

x = zeros(n,N);
z = zeros(n,N);
u = zeros(n,N);


if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\t%10s\t%10s\n', 'iter', ...
      'r norm', 'eps pri', 's norm', 'eps dual', 'objective');
end

for k = 1:MAX_ITER
    % prediction
    % u-update
    uu = u - rho*(x - z);
	% x-update
    for i = 1:N
        cvx_begin quiet
            variable x_var(n)
            minimize ( sum(pos(A{i}*x_var + 1)) + rho/2*sum_square(x_var -x(:,i)-uu(:,i)/rho) )
        cvx_end
        xx(:,i) = x_var;
    end
    xave = mean(xx,2);
    % z-update 
    zold = z;
    z = N*rho/(1/lambda + N*rho)*mean( xx-x+zold - uu/rho, 2 );
    zz = z*ones(1,N);
    % correction step
    u = u+rho*(x-xx+zz-z);
    x = x-nu*(x-xx+z-zz);
    z = z+nu*(zz-z);
    

    % diagnostics, reporting, termination checks
    history.objval(k)  = objective(A, lambda, p, x, z);
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

function obj = objective(A, lambda, p, x, z)
    obj = hinge_loss(A,x) + 1/(2*lambda)*sum_square(z(:,1));
end

function val = hinge_loss(A,x)
    val = 0;
    for i = 1:length(A)
        val = val + sum(pos(A{i}*x(:,i) + 1));
    end
end