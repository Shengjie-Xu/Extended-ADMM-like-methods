clc; 
clear;
close all;
%addpath ../Images                     %% ����ͼƬ����·��\�ļ���
%I = im2double(imread('shape.jpg'));   %% ��ͼƬ
%I = im2double(imread('chart.tiff'));
I = im2double(imread('housergb.png'));
rand('seed', 0);
randn('seed', 0);
[n1,n2,n3] = size(I);            %% ��ȡͼƬ��С
x0 = I+0.2*randn(n1,n2,n3);      %% �Ӹ�˹����
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
opts.beta=1;      %%% �㷨�ķ�����
opts.mu=5;        %%% ģ���еĲ���
opts.MaxIt=200;   %%% ��������
opts.Tol=1e-4;    %%% ͣ��׼��
[u,PSNR,Time] = TV_denoise(x0,opts,I);        %%% the original ADMM 
[u,PSNR,Time] = TV_denoise_pd_pc(x0,opts,I);  %%% the primal-dual extended ADMM
[u,PSNR,Time] = TV_denoise_dp_pc(x0,opts,I);  %%% the dual-primal extended ADMM
