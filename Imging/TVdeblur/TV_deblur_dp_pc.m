function [u,PSNR,Time,Itr] = TV_deblur_dp_pc(x0,h,opts,I)
% the dual-primal extended ADMM for image deblurring with total variation
nu=0.99;
[n1,n2,n3] = size(x0);
beta = opts.beta;
mu   = opts.mu;
MaxIt= opts.MaxIt;
Tol = opts.Tol;
%%%%%%%%%%% 周期边界条件：fourier域下的filter 
siz = size(h);
center = [fix(siz(1)/2+1),fix(siz(2)/2+1)];
P  = zeros(n1,n2,n3); 
for i =1:n3; P(1:siz(1),1:siz(2),i) = h; end
D  = fft2(circshift(P,1-center));
H  = @(x) real(ifft2(D.*fft2(x)));       %%%% Blur operator.  B x 
HT = @(x) real(ifft2(conj(D).*fft2(x))); %%%% Transpose of blur operator.

%%%%%%%%%%%%%%%%% 图像的梯度 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d1h = zeros(n1,n2,n3); d1h(1,1,:) = -1; d1h(n1,1,:) = 1; d1h = fft2(d1h);
d2h = zeros(n1,n2,n3); d2h(1,1,:) = -1; d2h(1,n2,:) = 1; d2h = fft2(d2h);
Px  = @(x) [x(2:n1,:,:)-x(1:n1-1,:,:); x(1,:,:)-x(n1,:,:)]; %%\nabla_1 x 
Py  = @(x) [x(:,2:n2,:)-x(:,1:n2-1,:), x(:,1,:)-x(:,n2,:)]; %%\nabla_2 y
PTx = @(x) [x(n1,:,:)-x(1,:,:); x(1:n1-1,:,:)-x(2:n1,:,:)]; %%\nabla_1^T x 
PTy = @(x) [x(:,n2,:)-x(:,1,:), x(:,1:n2-1,:)-x(:,2:n2,:)]; %%\nabla_2^T y
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Intionalization
u    = x0; 
v1   = Px(u);
v2   = Py(u);
lbd1= zeros(n1,n2,n3);
lbd2= zeros(n1,n2,n3);
vn1 = zeros(n1,n2,n3);
vn2 = zeros(n1,n2,n3);
MDu  = mu*abs(D).^2 + beta*(abs(d1h).^2+abs(d2h).^2); 
HTx0 = mu*HT(x0);
PSNR  = zeros(1,MaxIt);  
Time = cputime;

for k =1:MaxIt   
    % prediction
    % lambda-update
    lbd1nn = lbd1 - beta*(v1 - vn1);
    lbd2nn = lbd2 - beta*(v2 - vn2);  
    % u-update
    Temp= PTx(beta*v1+lbd1nn) + PTy(beta*v2+lbd2nn) + HTx0;
    un   = real(ifft2(fft2(Temp)./MDu));  
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
    if stopic<Tol
        k
        break
    end
    %%%%%%%%%停机准则，
    PSNR(k) = psnr(u,I);
    u=un;
    
end
Time = cputime-Time; Itr =k;

    

