% Demonstration script for the clustering algorithm
clear; clc;

datasets_folder = 'Chosen_datasets';
target_datasets = {'NGs'};

% Algorithm settings
params.max_iters = 100;    
params.k_ratio = 1.0;
params.k_neighbors = 25;
params.eta = 0.5608;

params.label_ratio = 0.1;
num_runs = 10;

for idx = 1:length(target_datasets)
    data_path = fullfile(datasets_folder, [target_datasets{idx} '.mat']);
    if ~exist(data_path, 'file'), continue; end
    load(data_path);

    n = length(gnd);
    nClass = length(unique(gnd));
    [gnd, ~] = grp2idx(gnd);

    X = cell(1, nviews);
    for v = 1:nviews
        if size(fea{v}, 1) == n, X{v} = fea{v}'; else, X{v} = fea{v}; end
        X{v} = double(max(X{v}, 0));
        X{v}(isnan(X{v})) = 0;
        X{v} = X{v} ./ (sqrt(sum(X{v}.^2, 1)) + 1e-10);
    end

    Y_ground_truth = full(sparse(1:n, gnd, 1, n, nClass));

    ACC_log = zeros(1, num_runs); 
    NMI_log = zeros(1, num_runs);
    Purity_log = zeros(1, num_runs);

    fprintf('Dataset: %s (Ratio: %.1f%%)\n', target_datasets{idx}, params.label_ratio*100);

    for run = 1:num_runs
        rand_indices = randperm(n);
        num_labeled = round(n * params.label_ratio);
        idx_labeled = rand_indices(1:num_labeled);
        idx_unlabeled = rand_indices(num_labeled+1:end);

        Y_input = Y_ground_truth;
        Y_input(idx_unlabeled, :) = 0;

        [W_full, D_full] = ConstructKNN_Graph(X, params.k_neighbors);
        k = round(nClass * params.k_ratio);

        tic;
        [Model, H_final, ~] = DPSSNMF(X, Y_input, k, params.max_iters, params, W_full, D_full);
            
        if ~any(isnan(H_final(:)))
            [res_acc, res_nmi, res_purity] = performance_kmeans(H_final, nClass, gnd);
            ACC_log(run) = res_acc(1);
            NMI_log(run) = res_nmi(1);
            Purity_log(run) = res_purity(1);
            fprintf('  ACC: %.4f | NMI: %.4f | Purity: %.4f \n', res_acc(1), res_nmi(1), res_purity(1));
        else
            fprintf('  Failed (NaN)\n');
        end
    end
    
    fprintf('Mean: ACC=%.4f | NMI=%.4f | Purity=%.4f\n', mean(ACC_log), mean(NMI_log), mean(Purity_log));
end