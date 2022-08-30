
n=500;
Tol=0.00001;
%%% generate the matrix C
rand('state',0);
C=rand(n,n); C=(C'+C)-ones(n,n)+eye(n);
%%% generate HL and HU
HU=ones(n,n)*0.2;
HL=-HU;
for i=1:n
    HU(i,i)=1;
    HL(i,i)=1;
end

cor_ADMM(n,C,HL,HU,Tol);   % the extended ADMM
cor_pd_pc(n,C,HL,HU,Tol);  % the primal-dual extended ADMM
cor_dp_pc(n,C,HL,HU,Tol);  % the dual-primal extended ADMM

