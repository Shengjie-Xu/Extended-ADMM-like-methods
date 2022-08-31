clc
clear
close all
addpath ../Images
I = im2double(imread('shape.jpg'));    %%%测试图片
I = im2double(imread('chart.tiff'));
 I = im2double(imread('housergb.png'));

[n1,n2,n3] = size(I);
h  = fspecial('aver',5);    %%%% filter
x0 = imfilter(I,h,'circular') + 0.20*randn(n1,n2,n3);  %%%加blur+noise

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.beta = 1; %%%罚参数
opts.mu   = 10; %%%模型参数
opts.MaxIt= 200;  %%%最大迭代次数
opts.Tol  = 1e-4; 
[u, PSNR, Time, Itr]  = TV_deblur(x0,h,opts,I);       % the original ADMM
[u, PSNR, Time, Itr]  = TV_deblur_pd_pc(x0,h,opts,I); % the primal-dual extended ADMM
[u, PSNR, Time, Itr]  = TV_deblur_dp_pc(x0,h,opts,I); % the dual-primal extended ADMM

