function w = apg_log_reg(X_p, X_n, rho, opts)
% uses apg to solve an L1-regularized logistic-regression problem
%
% min_w (sum_i( log(1 + exp(w'*x_i*y_i) ) + rho * norm(w,1))
%
% X_p is the matrix of positive instances (each column is one instance)
% X_n is the matrix of negative instances
% rho is the L1-regularization weight

    opts.X_p = X_p;
    opts.X_n = X_n;
    opts.N = size(X_p,2) + size(X_n,2);
    opts.rho = rho;
    w = apg(@lr_grad, @soft_thresh, size(X_p,1), opts);
    
end

function g = lr_grad(w, opts)
    v = exp(w'*[opts.X_p, -opts.X_n])';
    g = [opts.X_p, -opts.X_n]*(v./(1+v))/opts.N;
end

function v = soft_thresh(w, t, opts)
    v = sign(w) .* max(abs(w) - t*opts.rho,0);
end