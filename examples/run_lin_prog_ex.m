%% generate and solve linear program example
clear all; close all

%%
fprintf('running linear programming ex:\n');
density = 0.25;
n = 100; m = 300;

z = randn(m,1);
y = max(z,0); % y = s - z;
s = y - z; % s = proj_cone(z,K);

A = sprandn(m,n,density);
x = randn(n,1);
c = -A'*y;
b = A*x + s;

opts.EPS = 1e-7;
opts.ALPHA = 1;
opts.MAX_ITERS = 10000;

x_lp = apg_linear_program(A, b, c, opts);
title('linear program')

%% cvx for verification
try
    cvx_begin
    %cvx_solver 'scs'
    variables x_c(n) s(m)
    minimize(c'*x_c)
    A*x_c + s == b
    s >= 0
    cvx_end
catch
    disp('cvx failed (is cvx installed?)')
end