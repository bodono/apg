% this runs all the apg examples
clear all; %close all
addpath('..')

%% set optional parameters:
opts = [];
% for example:
% opts.QUIET = true;
% opts.GEN_PLOTS = false;
% opts.USE_RESTART = false;
% opts.ALPHA = 1;

%% lasso:
fprintf('running lasso ex:\n');
n = 1e3; m = 100; A = randn(m,n); b = randn(m,1); mu = 10;
x_lasso = apg_lasso(A, b, mu, opts);
title('lasso')

%% nonnegative QP:
fprintf('running nonnegative QP ex:\n');
n = 100; Q = randn(n); Q = Q*Q'; q = randn(n,1);
x_qp = apg_nonneg_qp(Q, q, opts);
title('nonnegative QP')

%% L1 regularized logistic regression:
fprintf('running L1 regularized logistic regression ex:\n');
n = 1e3; d = 250; rho=0.1; x_n = -0.1*rand(d,1); x_p = 0.1*rand(d,1);
X_p = 5*randn(d,n)+x_p*ones(1,n); X_n = 5*randn(d,n)+x_n*ones(1,n);
x_lr = apg_log_reg(X_p,X_n,rho,opts);
title('L1 regularized logistic regression')


%% noisy low-rank matrix completion
fprintf('running noisy low-rank matrix completion ex:\n');
n = 100; m = 50; r = 10; density = 0.2; rho = 5;
M = randn(m,r)*randn(r,n); mask = (rand(m,n)<density);
X_mc = apg_noisy_matrix_comp(M.*mask, rho, opts);
title('noisy low-rank matrix completion')
