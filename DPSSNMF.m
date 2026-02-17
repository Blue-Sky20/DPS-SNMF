function [Model, H_final, obj_history] = DPSSNMF(X, Y_label, k, max_iters, params, W_global, D_global)
    % Dual-Propagation Semi-Supervised Non-negative Matrix Factorization
    
    nviews = length(X);
    n = size(X{1}, 2);
    [~, c] = size(Y_label); 
    k = c;

    % Load parameters
    if isfield(params, 'k_neighbors'), k_neighbors = params.k_neighbors; else, k_neighbors = 5; end
    if isfield(params, 'power_p'), power_p = params.power_p; else, power_p = 2; end 
    if isfield(params, 'eta'), eta = params.eta; else, eta = 0.5; end
    
    % Data preprocessing
    is_labeled = sum(Y_label, 2) > 0;
    idx_labeled = find(is_labeled);
    idx_unlabeled = find(~is_labeled);
    num_labeled = length(idx_labeled);
    perm_idx = [idx_labeled; idx_unlabeled];
    
    X_sorted = cell(1, nviews);
    for v = 1:nviews
        X_sorted{v} = X{v}(:, perm_idx); 
    end
    Y_sorted = Y_label(perm_idx, :);
    W_global_sorted = W_global(perm_idx, perm_idx);
    D_global_sorted = D_global(perm_idx, perm_idx);
    
    % Construct KNN graphs for each view
    Ws_local = cell(1, nviews);
    for v = 1:nviews
        [Ws_local{v}, ~] = ConstructKNN_Graph({X_sorted{v}}, k_neighbors);
    end

    % Compute propagation matrix
    inv_D_diag = 1 ./ (diag(D_global_sorted) + 1e-10);
    P_prop = bsxfun(@times, inv_D_diag, W_global_sorted);

    % Initialize variables
    H_init = abs(randn(n, k));
    H_init(1:num_labeled, :) = Y_sorted(1:num_labeled, :); 
    H_init = H_init + 0.1; 
    H_init = H_init ./ sum(H_init, 2); 
    
    Hv_cell = cell(1, nviews);
    for v = 1:nviews
        Hv_cell{v} = H_init;
    end
    obj_history = zeros(1, max_iters);
    view_weights = ones(1, nviews) / nviews;

    % Optimization loop
    for iter = 1:max_iters
        
        % Step A: SNMF update
        for v = 1:nviews
            curr_Hv = Hv_cell{v}; 
            W_v = Ws_local{v};
            num_H = W_v * curr_Hv;
            den_H = curr_Hv * (curr_Hv' * curr_Hv);
            curr_Hv = curr_Hv .* (num_H ./ max(den_H, 1e-10));
            Hv_cell{v} = curr_Hv;
        end
        
        % Step B: View weighting
        rec_loss_list = zeros(1, nviews);
        for v = 1:nviews
            rec_diff = Ws_local{v} - Hv_cell{v} * Hv_cell{v}';
            rec_loss_list(v) = norm(rec_diff, 'fro')^2;
        end
        
        loss_scale_factor = mean(rec_loss_list) + 1e-10;
        scaled_loss = rec_loss_list / loss_scale_factor;
        scaled_loss_shifted = scaled_loss - min(scaled_loss);
        view_weights = exp(-scaled_loss_shifted); 
        view_weights = view_weights / sum(view_weights);
        
        % Step C: Compute temporary consensus
        H_consensus = zeros(n, k);
        for v = 1:nviews
            H_consensus = H_consensus + view_weights(v) * Hv_cell{v};
        end
        
        H_consensus(1:num_labeled, :) = Y_sorted(1:num_labeled, :);
        H_consensus = max(H_consensus, 0); 
        
        % Step D: Label propagation on consensus
        H_prop_direction = P_prop * H_consensus - H_consensus;
        H_unlabeled_old = H_consensus(num_labeled+1:end, :);
        H_unlabeled_direction = H_prop_direction(num_labeled+1:end, :);
        H_consensus(num_labeled+1:end, :) = H_unlabeled_old + eta * H_unlabeled_direction;

        % Step E: Injection, normalization, and sharpening
        for v = 1:nviews
            curr_H = Hv_cell{v};
            row_sums_v = sum(curr_H, 2);
            curr_H = bsxfun(@rdivide, curr_H, row_sums_v + 1e-10);
            
            row_sums_c = sum(H_consensus, 2);
            H_cons_norm = bsxfun(@rdivide, H_consensus, row_sums_c + 1e-10);

            H_temp = curr_H + H_cons_norm; 
            
            row_sums = sum(H_temp, 2);
            H_temp = bsxfun(@rdivide, H_temp, row_sums + 1e-10);
            
            H_sharp = H_temp .^ power_p;
            row_sums_s = sum(H_sharp, 2);
            H_temp = bsxfun(@rdivide, H_sharp, row_sums_s + 1e-10);
            
            Hv_cell{v} = H_temp;
        end
        
        obj_history(iter) = sum(rec_loss_list);
    end
    
    % Final fusion
    H_final_consensus = zeros(n, k);
    for v = 1:nviews
        H_final_consensus = H_final_consensus + view_weights(v) * Hv_cell{v};
    end

    H_final = zeros(n, k);
    H_final(perm_idx, :) = H_final_consensus;
    Model.B = eye(k); 
    Model.Z = [];
end