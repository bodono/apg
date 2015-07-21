function x = apg_linear_program(A, b, c, opts)
% uses apg to solve a linear problem
%
% min_(x,s) c'*x
% s.t.      Ax + s == b
%           s >= 0
%
    [m,n] = size(A);
    Q = sparse([zeros(n) A' c;
                -A zeros(m,m) b;
                -c' -b' 0]);
    
    opts.Q = Q;
    opts.n = n;
    opts.m = m;
    if ~isfield(opts,'X_INIT');opts.X_INIT = 10 * rand(2 * size(Q,2),1);end
    
    % uv = [x;y;tau;r;s;kap]
    uv = apg(@quad_grad, @proj_cone, 2 * size(Q,2), opts);
    
    x = uv(1:n) / uv(n+m+1);
end

function g = quad_grad(uv, opts)
    n = opts.n;
    m = opts.m;
    Qu = opts.Q*uv(1:n+m+1);
    v = uv(n+m+2:end);
    g = [opts.Q'*(Qu - v);v - Qu];
end

function uv = proj_cone(uv, ~, opts)
    n = opts.n;
    m = opts.m;
    uv(n+1:end) = max(uv(n+1:end),0); % y,s,tau,kap >= 0
    uv(n+m+2:2*n+m+2) = 0; % r = 0
end