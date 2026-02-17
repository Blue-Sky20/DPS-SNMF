function [CA, F, P, Recall, nmi, AR] = performance_kmeans(X, k, truth)
    % Evaluate clustering performance using k-means
    
    max_iter = 1000;
    replic = 20;

    if min(truth) == 0
        truth = truth + 1;
    end

    warning('off');
    
    n_samples = size(X, 1);
    if length(truth) ~= n_samples
        error('truth length (%d) does not match number of samples (%d)', length(truth), n_samples);
    end
    
    for i = 1:replic
        idx = kmeans(X, k, 'EmptyAction', 'singleton', 'maxiter', max_iter);
        
        if length(idx) ~= n_samples
            error('K-means returned %d labels, expected %d', length(idx), n_samples);
        end
        
        CAi(i) = 1 - compute_CE(idx, truth);
        [Fi(i), Pi(i), Ri(i)] = compute_f(truth, idx);
        nmii(i) = compute_nmi(truth, idx);
        ARi(i) = rand_index(truth, idx);
    end
    
    CA(1) = mean(CAi); CA(2) = std(CAi);
    F(1) = mean(Fi); F(2) = std(Fi);
    P(1) = mean(Pi); P(2) = std(Pi);
    Recall(1) = mean(Ri); Recall(2) = std(Ri);
    nmi(1) = mean(nmii); nmi(2) = std(nmii);
    AR(1) = mean(ARi); AR(2) = std(ARi);
end