function x = apg_lasso(A, b, rho, opts)
% uses apg to solve a lasso problem
%
% min_x (1/2)*sum_square(A*x - b) + rho * norm(x,1) 
%
% rho is the L1-regularization weight
    opts.A = A;
    opts.b = b;
    opts.rho = rho;
    x = apg(@quad_grad, @soft_thresh, size(A,2), opts);
end

function g = quad_grad(x, opts)
    g = opts.A'*(opts.A*x - opts.b);
end

function v = soft_thresh(x, t, opts)
    v = sign(x) .* max(abs(x) - t*opts.rho,0);
end