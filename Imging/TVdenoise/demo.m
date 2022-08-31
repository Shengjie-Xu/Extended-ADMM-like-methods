clc; 
clear;
close all;
%addpath ../Images                     %% 加入图片所在路径\文件夹
%I = im2double(imread('shape.jpg'));   %% 读图片
%I = im2double(imread('chart.tiff'));
I = im2double(imread('housergb.png'));
rand('seed', 0);
randn('seed', 0);
[n1,n2,n3] = size(I);            %% 获取图片大小
x0 = I+0.2*randn(n1,n2,n3);      %% 加高斯噪声
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.beta=1;      %%% 算法的罚参数
opts.mu=5;        %%% 模型中的参数
opts.MaxIt=200;   %%% 迭代次数
opts.Tol=1e-4;    %%% 停机准则
[u,PSNR,Time] = TV_denoise(x0,opts,I);        %%% the original ADMM 
[u,PSNR,Time] = TV_denoise_pd_pc(x0,opts,I);  %%% the primal-dual extended ADMM
[u,PSNR,Time] = TV_denoise_dp_pc(x0,opts,I);  %%% the dual-primal extended ADMM
