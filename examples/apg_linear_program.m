function x = apg_linear_program(A, b, c, opts)
% uses apg to solve a linear problem
%
% min_x     c'*x
% s.t.      Ax <= b
%
    [m,n] = size(A);
    % uses HSD embedding
    Q = sparse([zeros(n) A' c;
                -A zeros(m,m) b;
                -c' -b' 0]);

    opts.Q = Q;
    opts.n = n;
    opts.m = m;
    % cannot initialize to zero, since zero is a degenerate solution to HSD
    if ~isfield(opts,'X_INIT');
        opts.X_INIT = zeros(2 * size(Q,2),1);
        opts.X_INIT(n+m+1) = 10; % just set tau to non-zero
    end

    % uv = [x;y;tau;r;s;kap]
    uv = apg(@quad_grad, @proj_cone, 2 * size(Q,2), opts);

    x = uv(1:n) / uv(n+m+1); % assumes feasibility
    % the HSD embedding can detect infeasibility and return
    % certificates, this can easily be added to this later
end

function g = quad_grad(uv, opts)
    n = opts.n;
    m = opts.m;
    Qu_v = opts.Q*uv(1:n+m+1) - uv(n+m+2:end);
    g = [opts.Q'*(Qu_v);-Qu_v];
end

function uv = proj_cone(uv, ~, opts)
    n = opts.n;
    m = opts.m;
    uv(n+1:end) = max(uv(n+1:end),0); % y,s,tau,kap >= 0
    uv(n+m+2:2*n+m+2) = 0; % r = 0
end