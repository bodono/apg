apg v0.1b (@author bodonoghue)

Implements an Accelerated Proximal Gradient method
(Nesterov 2007, Beck and Teboulle 2009)

solves: 

    minimize f(x) + h(x)
    over x \in R^dim_x

where f is smooth, convex and h is non-smooth, convex but simple
in that we can easily evaluate the proximal operator of h

this takes in two function handles:
    f_grad(v,options) = df(v)/dv (gradient of f)
    prox_h(v,t,options) = argmin_x (t*h(x) + 1/2 * norm(x-v)^2)
         (where t is the step size at that iteration)

    if h = 0, then use prox_h = [] or prox_h = @(x,t,options)(x)

put all necessary function data in options fields

implements something similar to TFOCS step-size adaptation (Becker, Candes and Grant 2010)
and gradient-scheme adaptive restarting (O'Donoghue and Candes 2013)

quits when norm(y(k) - x(k+1)) < EPS

see examples/ for instances

optional options fields defined are below (with defaults)
to use defaults simply call apg with options = []

    X_INIT = zeros(dim_x,1); % initial starting point

    USE_RESTART = true; % use adaptive restart scheme

    MAX_ITERS = 2000; % maximum iterations before termination

    EPS = 1e-6; % tolerance for termination

    ALPHA = 1.01; % step-size growth factor

    BETA = 0.5; % step-size shrinkage factor

    QUIET = false; % writes out iter number every 100 iters

    GEN_PLOTS = true; % generate plots of function and gradient values

    USE_GRA = false; % use unaccelerated proximal gradient descent
