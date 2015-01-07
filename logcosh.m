% log cosh penalty
% Niru Maheswaranathan
% 03:01 PM Jun 17, 2014

function [f, grad] = logcosh(w, x, lambda, alpha)

    [k, n] = size(x);

    W = reshape(w, k, k);
    f = mean(sum(log(cosh(alpha*W*x))/alpha) + (0.5*lambda)*norms(W'*W*x - x));

    z = W*(x*x');
    u = tanh(alpha*W*x)*x';
    grad = vec(u + lambda * ( (W*W')*z + z*(W'*W) - z ))/n;

end
