%Apply features normalization with pre-given mu and sigma

function [X_norm, mu, sigma] = featureAddNormalize(X, m, s)

mu = m;
sigma = s;

X_norm = (X - mu) ./sigma;

end