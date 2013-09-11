% this runs all the apg examples
clear all; close all
randn('seed',sum('apg_examples1'));
randn('seed',sum('apg_examples2'));
addpath('..')

%% set optional parameters:
options = [];
%options.QUIET = true;
%options.GEN_PLOTS = false;
%options.USE_RESTART = false;

%% lasso:
n = 1e3; m = 100; A = randn(m,n); b = randn(m,1); mu = 10;
x_lasso = apg_lasso(A, b, mu, options);

%% nonnegative QP:
n = 100; Q = randn(n); Q = Q*Q'; q = randn(n,1);
x_qp = apg_nonneg_qp(Q, q, options);

%% L1 regularized logistic regression:
n = 1e3; d = 250; rho=0.1; x_n = -0.1*rand(d,1); x_p = 0.1*rand(d,1);
X_p = 5*randn(d,n)+x_p*ones(1,n); X_n = 5*randn(d,n)+x_n*ones(1,n);
x_lr = apg_log_reg(X_p,X_n,rho,options);