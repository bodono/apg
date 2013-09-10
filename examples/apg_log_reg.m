function w = apg_log_reg(X_p, X_n, rho, options)
% uses apg to solve an L1-regularized logistic-regression problem
%
% min_w (sum_i( log(1 + exp(w'*x_i*y_i) ) + rho * norm(w,1))
%
% X_p is the matrix of positive instances (each column is one instance)
% X_n is the matrix of negative instances
% rho is the L1-regularization weight

    options.X_p = X_p;
    options.X_n = X_n;
    options.N = size(X_p,2) + size(X_n,2);
    options.rho = rho;
    w = apg(@lr_grad, @soft_thresh, size(X_p,1), options);
    
end

function g = lr_grad(w, o)
    v = exp(w'*[o.X_p, -o.X_n])';
    g = [o.X_p, -o.X_n]*(v./(1+v))/o.N;
end

function v = soft_thresh(w, t, o)
    v = sign(w) .* max(abs(w) - t*o.rho,0);
end