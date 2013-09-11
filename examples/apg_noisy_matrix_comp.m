function L = apg_noisy_matrix_comp(M, rho, options)
%
%   min_X (1/2) sum_ij (L_ij - M_ij)^2 + rho * norm_nuc(L)
%
%   where the sum is over the known entries in M 
%   (the non-zero entries in this simple example)

    options.M = M;
    options.rho = rho;
    l = apg(@f_grad, @svd_shrink, size(M,1)*size(M,2), options);
    L = reshape(l,size(M));
end

function g = f_grad(x, o)
    L = reshape(x,size(o.M));
    G = (L-o.M).*(o.M ~= 0);
    g = G(:);
end

function v = svd_shrink(x, t, o)
    L = reshape(x,size(o.M));
    [U,S,V] = svd(L,'econ');
    s = diag(S);
    S = diag(sign(s) .* max(abs(s) - t*o.rho,0));
    L = U*S*V';
    v = L(:);
end