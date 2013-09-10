function x = apg_nonneg_qp(Q, q, options)
% uses apg to solve a nonnegative QP
%
% min_x (1/2)*x'*Q*x +q'*x
% s.t.  x>=0
%
    options.Q = Q;
    options.q = q;
    f_grad = @(x,o)(o.Q*x - o.q);
    prox_h = @(x,t,o)(pos(x));
    x = apg(f_grad, prox_h, size(q,1), options);
end