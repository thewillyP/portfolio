function [X_norm, mu, sigma] = featureNormalize(X)

% Initialize
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% Solve
mu = mean(X_norm);
X_norm = X - mu;
sigma = std(X_norm);
X_norm = X_norm ./sigma;


end
