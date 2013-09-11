function L = apg_noisy_matrix_comp(M, rho, opts)
%   this reconstructs a low rank matrix   
%   given noisy incomplete measurements, solves: 
%
%   min_X (1/2) sum_ij (L_ij - M_ij)^2 + rho * norm_nuc(L)
%
%   where the sum is over the known entries in M 
%   (the non-zero entries in this simple example)
    opts.M = M;
    opts.rho = rho;
    l = apg(@grad_f, @svd_shrink, size(M,1)*size(M,2), opts);
    L = reshape(l,size(M));
end

function g = grad_f(x, opts)
    L = reshape(x,size(opts.M));
    G = (L-opts.M).*(opts.M ~= 0);
    g = G(:);
end

function v = svd_shrink(x, t, opts)
    L = reshape(x,size(opts.M));
    [U,S,V] = svd(L,'econ');
    s = diag(S);
    S = diag(sign(s) .* max(abs(s) - t*opts.rho,0));
    L = U*S*V';
    v = L(:);
end