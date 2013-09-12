apg 
=====================
(@author bodonoghue) MATLAB script  
Implements an Accelerated Proximal Gradient method
(Nesterov 2007, Beck and Teboulle 2009)

solves: 

    minimize f(x) + h(x)
    over x \in R^dim_x

where `f` is smooth, convex - user supplies function to evaluate gradient of `f`  
`h` is convex - user supplies function to evaluate the proximal operator of `h`

call as:

    x = apg( grad_f, prox_h, dim_x, opts )

this takes in two function handles:

    grad_f(v, opts) = df(v)/dv 
    (gradient of f at v)
    
    prox_h(v, t, opts) = argmin_x ( t * h(x) + 1/2 * norm(x-v)^2 )
    where t is the (scalar, positive) step size at that iteration

    if h = 0, then use prox_h = [] or prox_h = @(x,t,opts)(x)

put all necessary function data in opts struct 

each iteration of `apg` requires one gradient evaluation of `f` and one prox step with `h`

quits when:
    
    norm( y(k) - x(k+1) ) < EPS * max( 1,norm( x(k+1) )

`apg` implements something similar to TFOCS step-size adaptation (Becker, Candes and Grant 2010)  
and gradient-scheme adaptive restarting ([O'Donoghue and Candes 2013](http://bodonoghue.org/publications/adap_restart.pdf))

see examples/ for usage instances

optional opts fields defined are below (with defaults)  
to use defaults simply call apg with `opts = []`

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
