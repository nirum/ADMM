% RICA
% Niru Maheswaranathan
% 02:35 PM Jun 17, 2014
addtoolbox('minFunc');
addtoolbox('cvx');
cvx_startup;
setpaths_minFunc;

k = 5;
n = 1000;

% generate data
W = orth(randn(k));
s = laprnd(k,n);
x = W \ s;

lambda = 2;

%cvx_begin
    %variable What(k,k)
    %minimize norm(What*x,1) + lambda*sum(norms(What'*What*x - x))
%cvx_end

phi = @(x) log(cosh(x));

alpha = 1;
fobj = @(w) logcosh(w, x, lambda, alpha);
options = struct('Method', 'qnewton', 'Display', 'iter', 'MaxIter', 1000, 'MaxFunEvals', 5000, 'numDiff', 1);

W0 = vec(randn(k));
[What, objval, exitflag, output] = minFunc(fobj, W0, options);
What = reshape(What, k, k);
shat = What*x;
