function x = apg_lasso(A, b, rho, opts)
% uses apg to solve a lasso problem
%
% min_x (1/2)*sum_square(A*x - b) + rho * norm(x,1) 
%
% rho is the L1-regularization weight
    opts.A = A;
    opts.b = b;
    opts.rho = rho;
    opts.dims = [size(A,2), size(b,2)];
    x = apg(@quad_grad, @soft_thresh, prod(opts.dims), opts);
end

function g = quad_grad(x, opts)
    x_t = reshape(x, opts.dims);
    g = opts.A'*(opts.A*x_t - opts.b);
    g = g(:);
end

function v = soft_thresh(x, t, opts)
    v = sign(x) .* max(abs(x) - t*opts.rho,0);
end