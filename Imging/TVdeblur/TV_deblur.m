
function [u,PSNR,Time,Itr] = TV_deblur(x0,h,opts,I)

% the original ADMM for image deblurring with total variation
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
lbd11= zeros(n1,n2,n3);
lbd12= zeros(n1,n2,n3);
MDu  = mu*abs(D).^2 + beta*(abs(d1h).^2+abs(d2h).^2);
HTx0 = mu*HT(x0);
PSNR  = zeros(1,MaxIt);  
Time = cputime;

for k =1:MaxIt    
    %%% step 1: u -子问题
    Temp= PTx(beta*v1+lbd11) + PTy(beta*v2+lbd12) + HTx0;
    un   = real(ifft2(fft2(Temp)./MDu));
    
    %%% step 2: v-子问题
    sk1 = Px(un) - lbd11/beta;
    sk2 = Py(un) - lbd12/beta;
    nsk = sqrt(sk1.^2 + sk2.^2); nsk(nsk==0)=1;
    nsk = max(1-1./(beta*nsk),0);
    v1 = sk1.*nsk;
    v2 = sk2.*nsk;
         
    %%%% 更新 lambda
    lbd11 = lbd11 - beta*(Px(un) - v1);
    lbd12 = lbd12 - beta*(Py(un) - v2);
    stopic=norm(u(:)-un(:))/norm(un(:));
    if stopic<Tol
        k
        break
    end
    %%%%%%%%%停机准则，
    PSNR(k) = psnr(u,I);
    u=un;
    
end
Time = cputime-Time; Itr =k;

    

