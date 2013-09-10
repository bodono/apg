function x = apg(f_grad, prox_h, dim_x, options)
%
% apg v0.1b (@author bodonoghue)
%
% Implements an Accelerated Proximal Gradient method
% (Nesterov 2007, Beck and Teboulle 2009)
%
% solves: min_x (f(x) + h(x)), x \in R^dim_x
%
% where f is smooth, convex and h is non-smooth, convex but simple
% in that we can easily evaluate the proximal operator of h
%
% this takes in two function handles:
% f_grad(v,options) = df(v)/dv (gradient of f)
% prox_h(v,t,options) = argmin_x (t*h(x) + 1/2 * norm(x-v)^2)
%                       where t is the step size at that iteration
% if h = 0, then use prox_h = [] or prox_h = @(x,t,options)(x)
% put the necessary function data in options fields
%
% implements something similar to TFOCS step-size adaptation (Becker, Candes and Grant 2010)
% and gradient-scheme adaptive restarting (O'Donoghue and Candes 2013)
%
% quits when norm(y(k) - x(k+1)) < EPS * max(1, norm(x(k+1))
%
% optional options fields defined are below (with defaults)
% to use defaults simply call apg with options = []
X_INIT = zeros(dim_x,1); % initial starting point
USE_RESTART = true; % use adaptive restart scheme
MAX_ITERS = 2000; % maximum iterations before termination
EPS = 1e-6; % tolerance for termination
ALPHA = 1.01; % step-size growth factor
BETA = 0.5; % step-size shrinkage factor
QUIET = false; % writes out iter number every 100 iters
GEN_PLOTS = true; % generate plots of function and gradient values
USE_GRA = false; % use unaccelerated proximal gradient descent

if (~isempty(options))
    if isfield(options,'X_INIT');X_INIT = options.X_INIT;end
    if isfield(options,'USE_RESTART');USE_RESTART = options.USE_RESTART;end
    if isfield(options,'MAX_ITERS');MAX_ITERS = options.MAX_ITERS;end
    if isfield(options,'EPS');EPS = options.EPS;end
    if isfield(options,'ALPHA');ALPHA = options.ALPHA;end
    if isfield(options,'BETA');BETA = options.BETA;end
    if isfield(options,'QUIET');QUIET = options.QUIET;end
    if isfield(options,'GEN_PLOTS');GEN_PLOTS = options.GEN_PLOTS;end
    if isfield(options,'USE_GRA');USE_GRA = options.USE_GRA;end
end

if (GEN_PLOTS); errs = zeros(MAX_ITERS,2);end

x = X_INIT; y=x;
g = f_grad(y,options);
theta = 1;

% perturbation for first step-size estimate:
x_hat = x + ones(dim_x,1);
t = norm(x - x_hat)/norm(g - f_grad(x_hat,options));

for k=1:MAX_ITERS
    
    if (~QUIET && mod(k,100)==0)
        fprintf('iter num %i, norm(tGk): %1.2e, step-size: %1.2e\n',k,err1,t);
    end
    
    x_old = x;
    y_old = y;
    
    x = y - t*g;
    
    if ~isempty(prox_h)
        x = prox_h(x,t,options);
    end
    
    err1 = norm(y-x)/max(1,norm(x));
    err2 = norm(x-x_old)/max(1,norm(x));
    
    if (GEN_PLOTS);
        errs(k,1) = err1; 
        errs(k,2) = err2;
    end
    
    if (err1 < EPS)
        break;
    end
    
    if(~USE_GRA)
        theta = 2/(1 + sqrt(1+4/(theta^2)));
    else
        theta = 1;
    end
    
    if (USE_RESTART && (y_old-x)'*(x-x_old)>0)
        x = x_old;
        y = x;
        theta = 1;
    else
        y = x + (1-theta)*(x-x_old);
    end
    
    g_old = g;
    g = f_grad(y,options);
    
    % TFOCS-style backtracking:
    t_hat = 0.5*(norm(y-y_old)^2)/abs((y - y_old)'*(g_old - g));
    t = min( ALPHA*t, max( BETA*t, t_hat ) );
    
end
if (~QUIET)
    fprintf('iter num %i, norm(tGk): %1.2e, step-size: %1.2e\n',k,err1,t);
    fprintf('Terminated\n');
end
if (GEN_PLOTS)
    errs = errs(1:k,:);
    figure();semilogy(errs(:,1));
    xlabel('iters');title('norm(tGk)')
    figure();semilogy(errs(:,2));
    xlabel('iters');title('norm(Dxk)')
end

end