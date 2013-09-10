function x = apg_lasso(A, b, rho, options)
% uses apg to solve a lasso problem
%
% min_x (1/2)*sum_square(A*x - b) + rho * norm(x,1) 
%
% rho is the L1-regularization weight
    options.A = A;
    options.b = b;
    options.rho = rho;
    x = apg(@quad_grad, @soft_thresh, size(A,2), options);
end

function g = quad_grad(x, o)
    g = o.A'*(o.A*x - o.b);
end

function v = soft_thresh(x, t, o)
    v = sign(x) .* max(abs(x) - t*o.rho,0);
end