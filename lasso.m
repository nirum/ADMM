% ADMM Lasso example
% Niru Maheswaranathan
% 4/23/14

% generate a problem
n = 150;                             % measurements
p = 500;                             % regressors
A = randn(n,p);                      % sensing matrix
x0 = randn(p,1).*(rand(p,1) < 0.05); % signal (sparse)
b = A*x0;                            % measurement
lambda = 0.1;                        % sparsity penalty
rho = 0.05;                          % augmented Lagrange parameter

% initialize
numiter = 100;
x = randn(p,1);
z = randn(p,1);
y = zeros(p,1);
err = zeros(numiter,1);

% caching
P = eye(p)/rho - (A' * pinv(rho * eye(n) + A * A') * A)/rho;
c = P*A'*b;

% ADMM
for k = 1:numiter

    % updates
    x = c + P*(rho*z - y);
    z = wthresh(x + y / rho, 's', lambda / rho);
    y = y + rho*(x - z);

    % store error
    progressbar(k,numiter);
    err(k) = norm(x-x0) / p;

end
