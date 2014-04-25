% testing speedup using the matrix inversion lemma
% Niru Maheswaranathan
% Apr 25 2014

% test different parameter sizes
nvals = round(logspace(1,3,25));
p = max(nvals);

% initialize variables
err   = zeros(size(nvals));
naive = zeros(size(nvals));
lemma = zeros(size(nvals));

for idx = 1:length(nvals)

    % generate a matrix
    n = nvals(idx);
    A = randn(n,p);

    % naive inverse
    tic; P1 = pinv(A'*A + rho*eye(p)); naive(idx) = toc;

    % matrix inversion lemma
    tic; P2 = eye(p)/rho - (A' * pinv(rho * eye(n) + A * A') * A)/rho; lemma(idx) = toc;

    % store error
    err(idx) = norm(P1-P2,'fro');
    progressbar(idx,length(nvals));

end

% plots
fig(1);
loglog(nvals/p, naive, 'ko-', nvals/p, lemma, 'ro-');
xlabel('Fracitonal rank (n/p)', 'FontSize', 24);
ylabel('Time to compute inverse (seconds)', 'FontSize', 24);
title('Matrix inversion via the Sherman-Morrison-Woodbury identity', 'FontSize', 30);
legend('Naive', 'Woodbury', 'Location', 'SouthEast');
makepretty; grid on;

fig(2);
semilogx(nvals/p, err, 'bo-');
xlabel('Fracitonal rank (n/p)', 'FontSize', 24);
ylabel('Fro. norm distance between the estimates', 'FontSize', 24)
title('Discrepancy between Woodbury and naive inverse', 'FontSize', 30);
makepretty; grid on;
