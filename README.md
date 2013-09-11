apg v0.1b (@author bodonoghue) MATLAB script

Implements an Accelerated Proximal Gradient method
(Nesterov 2007, Beck and Teboulle 2009)

solves: 

    minimize f(x) + h(x)
    over x \in R^dim_x

where f is smooth, convex and h is non-smooth, convex but simple
in that we can easily evaluate the proximal operator of h

call as:

    x = apg( f_grad, prox_h, dim_x, options )

this takes in two function handles:

    f_grad(v,options) = df(v)/dv 
    (gradient of f at v)
    
    prox_h(v,t,options) = argmin_x ( t*h(x) + 1/2 * norm(x-v)^2 )
    (where t is the step size at that iteration)

    if h = 0, then use prox_h = [] or prox_h = @(x,t,options)(x)

put all necessary function data in options fields

implements something similar to TFOCS step-size adaptation (Becker, Candes and Grant 2010)
and gradient-scheme adaptive restarting (O'Donoghue and Candes 2013)

quits when:
    
    norm( y(k) - x(k+1) ) < EPS * max( 1,norm( x(k+1) )

see examples/ for usage instances

optional options fields defined are below (with defaults)
to use defaults simply call apg with options = []

    X_INIT = zeros(dim_x,1); % initial starting point

    USE_RESTART = true; % use adaptive restart scheme

    MAX_ITERS = 2000; % maximum iterations before termination

    EPS = 1e-6; % tolerance for termination

    ALPHA = 1.01; % step-size growth factor

    BETA = 0.5; % step-size shrinkage factor

    QUIET = false; % if false writes out information every 100 iters

    GEN_PLOTS = true; % if true generates plots of proximal gradient

    USE_GRA = false; % if true uses unaccelerated proximal gradient descent

Example of usage:

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
