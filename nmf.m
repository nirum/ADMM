% non-negative matrix factorization
% Niru Maheswaranathan
% 11:41 AM Jun 17, 2014

% toy data
n = 2; m = 4; k = 3;
X = rand(n,k);
Y = rand(m,k);
A = X*Y';

% initialize
Xhat = randn(n,k);
Yhat = randn(m,k);
Ux = rand(n,k);
Uy = rand(m,k);
Vx = rand(n,k);
Vy = rand(m,k);

% parameters
lambda = 0.0001;
numiter = 1e4;
err = zeros(numiter,1);

% run admm
for j = 1:numiter

    Xhat = (A*Yhat + lambda*(Ux-Vx)) * pinv(Yhat'*Yhat + lambda*eye(k));
    Yhat = (pinv(Xhat'*Xhat + lambda*eye(k)) * (Xhat'*A + lambda*(Uy-Vy)'))';

    Ux = max(Ux,0);
    Uy = max(Uy,0);

    Vx = Vx - lambda*(Xhat - Ux);
    Vy = Vy - lambda*(Yhat - Uy);

    err(j) = norm(X-Xhat, 'fro') + norm(Y-Yhat, 'fro');

end
