function [u,PSNR,Time,Itr] = TV_denoise_dp_pc(x0,opts,I)
%%%% This function solve the unconstrained model for image deblurring
%%%% min_u |\nabla u|_1 + 0.5mu|u-u0|_2^2
%%%%  u -- original image
%%%%  u0 -- observed image
%%%% the dual-primal extended ADMM

nu=0.99;
[n1,n2,n3] = size(x0);  %% 图片大小
beta = opts.beta;       %% 罚参数
mu   = opts.mu;         %% 模型参数
MaxIt= opts.MaxIt;      %% 最大迭代次数
Tol=opts.Tol;
%%%%%%%%%%%%%%%%% 周期边界条件 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1h = zeros(n1,n2,n3); d1h(1,1,:) = -1; d1h(n1,1,:) = 1; d1h = fft2(d1h);
d2h = zeros(n1,n2,n3); d2h(1,1,:) = -1; d2h(1,n2,:) = 1; d2h = fft2(d2h);
Px  = @(x) [x(2:n1,:,:)-x(1:n1-1,:,:); x(1,:,:)-x(n1,:,:)]; %%\nabla_1 x 
Py  = @(x) [x(:,2:n2,:)-x(:,1:n2-1,:), x(:,1,:)-x(:,n2,:)]; %%\nabla_2 y
PTx = @(x) [x(n1,:,:)-x(1,:,:); x(1:n1-1,:,:)-x(2:n1,:,:)]; %%\nabla_1^T x 
PTy = @(x) [x(:,n2,:)-x(:,1,:), x(:,1:n2-1,:)-x(:,2:n2,:)]; %%\nabla_2^T y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%Intionalization
u    = x0; 
v1   = Px(u);
v2   = Py(u);
lbd1 = zeros(n1,n2,n3);
lbd2 = zeros(n1,n2,n3);
vn1 = zeros(n1,n2,n3);
vn2 = zeros(n1,n2,n3);
MDu  = mu + beta*(abs(d1h).^2+abs(d2h).^2);  %%%线性方程组
HTx0 = mu*x0;
PSNR = zeros(1,MaxIt); 

Time = cputime;  %%%记录时间
for k = 1:MaxIt 
    
    % prediction
    % lambda-update
    lbd1nn = lbd1 - beta*(v1 - vn1);
    lbd2nn = lbd2 - beta*(v2 - vn2);  
    % u-update
    Temp= PTx(beta*v1+lbd1nn) + PTy(beta*v2+lbd2nn) + HTx0;
    un  = real(ifft2(fft2(Temp)./MDu));   
    stopic=norm(u(:)-un(:))/norm(un(:));
    u=un;
    % v-update
    v11= Px(un);
    v22= Py(un);
    sk1 = vn1+v11-v1 - lbd1nn/beta;
    sk2 = vn2+v22-v2 - lbd2nn/beta;
    nsk = sqrt(sk1.^2 + sk2.^2); nsk(nsk==0)=1;
    nsk = max(1-1./(beta*nsk),0);
    vn1n = sk1.*nsk;
    vn2n = sk2.*nsk;         
    
    % correction 
    lbd1 = lbd1nn + beta*(v1-v11+vn1n-vn1);
    lbd2 = lbd2nn + beta*(v2-v22+vn2n-vn2);
    v1   = v1-nu*(v1-v11+vn1-vn1n);
    v2   = v2-nu*(v2-v22+vn2-vn2n);
    vn1  = vn1+nu*(vn1n-vn1);
    vn2  = vn2+nu*(vn2n-vn2);
    %stopic = norm(u(:)-un(:))/norm(un(:));
    if k>30 && stopic<Tol
        k
        break
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%
    PSNR(k) = psnr(u,I);  %%% 计算信噪比
    Time = cputime - Time;


end
Time = cputime-Time; Itr =k;

    

