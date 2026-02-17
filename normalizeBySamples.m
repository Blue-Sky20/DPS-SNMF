function X_norm = normalizeBySamples(X)
    % Normalize each sample feature vector to unit norm
    
    norms = sqrt(sum(X.^2, 1));
    norms(norms == 0) = 1; 
    X_norm = X ./ norms;
end