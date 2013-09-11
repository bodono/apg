function x = apg_nonneg_qp(Q, q, opts)
% uses apg to solve a nonnegative QP
%
% min_x (1/2)*x'*Q*x +q'*x
% s.t.  x>=0
%
    opts.Q = Q;
    opts.q = q;
    grad_f = @(x,opts)(opts.Q*x - opts.q);
    prox_h = @(x,t,opts)(pos(x));
    x = apg(grad_f, prox_h, size(q,1), opts);
end