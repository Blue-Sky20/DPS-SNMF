function [X_shuffled, labeled_mask, gnd_shuffled] = randpermData(X, gnd, C, n, l)
    % Shuffle data and allocate labels evenly across classes
    
    assert(length(gnd) == n, 'Label count mismatch with sample count');
    for v = 1:length(X)
        assert(size(X{v}, 2) == n, 'Sample count mismatch in view %d', v);
    end
    
    % Count samples per class
    class_indices = cell(1, C);
    class_counts = zeros(1, C);
    for c = 1:C
        class_indices{c} = find(gnd == c);
        class_counts(c) = length(class_indices{c});
    end
    
    % Allocate base number of labels per class
    base_num = floor(l / C);  
    remainder = l - base_num * C;  
    
    labeled_indices = [];
    
    for c = 1:C
        if class_counts(c) == 0
            warning('Class %d is empty, skipping label allocation', c);
            continue;
        end
        
        num_to_allocate = min(base_num, class_counts(c));
        if num_to_allocate > 0
            rand_order = randperm(class_counts(c));
            selected = class_indices{c}(rand_order(1:num_to_allocate));
            labeled_indices = [labeled_indices; selected(:)];
        end
    end
    
    % Allocate remainder labels to available classes
    if remainder > 0
        available_classes = [];
        for c = 1:C
            current_allocated = sum(ismember(labeled_indices, class_indices{c}));
            if class_counts(c) > current_allocated
                available_classes = [available_classes, c];
            end
        end
        
        if ~isempty(available_classes)
            for i = 1:min(remainder, length(available_classes))
                c = available_classes(i);
                current_allocated = sum(ismember(labeled_indices, class_indices{c}));
                remaining_in_class = class_counts(c) - current_allocated;
                
                if remaining_in_class > 0
                    already_selected = intersect(labeled_indices, class_indices{c});
                    remaining_indices = setdiff(class_indices{c}, already_selected);
                    
                    new_selected = remaining_indices(randperm(length(remaining_indices), 1));
                    labeled_indices = [labeled_indices; new_selected(:)];
                end
            end
        end
    end
    
    % Supplement labels from larger classes if total is still insufficient
    current_l = length(labeled_indices);
    if current_l < l
        needed = l - current_l;
        [~, size_order] = sort(class_counts, 'descend');
        for c = size_order
            if needed <= 0, break; end
            if class_counts(c) == 0, continue; end
            
            current_allocated = sum(ismember(labeled_indices, class_indices{c}));
            remaining_in_class = class_counts(c) - current_allocated;
            
            if remaining_in_class > 0
                to_allocate = min(remaining_in_class, needed);
                already_selected = intersect(labeled_indices, class_indices{c});
                remaining_indices = setdiff(class_indices{c}, already_selected);
                
                new_selected = remaining_indices(randperm(length(remaining_indices), to_allocate));
                labeled_indices = [labeled_indices; new_selected(:)];
                needed = needed - to_allocate;
            end
        end
    end
    
    % Construct global index (labeled first, then unlabeled)
    unlabeled_indices = setdiff(1:n, labeled_indices);
    
    labeled_indices = labeled_indices(randperm(length(labeled_indices)));
    unlabeled_indices = unlabeled_indices(randperm(length(unlabeled_indices)));
    
    global_indices = [labeled_indices(:); unlabeled_indices(:)]';
    
    % Generate label mask
    labeled_mask = zeros(1, n);
    labeled_mask(1:l) = 1;
    
    % Shuffle data and ground truth
    gnd_shuffled = gnd(global_indices);
    X_shuffled = cell(size(X));
    for v = 1:length(X)
        X_shuffled{v} = X{v}(:, global_indices);
    end
    
    % Print allocation statistics
    fprintf('\nLabel Allocation Stats (Total=%d, First %d samples):\n', l, l);
    fprintf('%-8s %-12s %-12s %-10s\n', 'Class', 'Total', 'Labeled', 'Ratio');
    
    allocated_per_class = zeros(1, C);
    for c = 1:C
        if class_counts(c) > 0
            class_labeled = sum(gnd_shuffled(1:l) == c);
            allocated_per_class(c) = class_labeled;
            fprintf('%-8d %-12d %-12d %-10.1f%%\n',...
                    c, class_counts(c), class_labeled,...
                    class_labeled/class_counts(c)*100);
        else
            fprintf('%-8d %-12d %-12d %-10s\n', c, 0, 0, 'N/A');
        end
    end
    
    valid_allocations = allocated_per_class(allocated_per_class > 0);
    if ~isempty(valid_allocations)
        fprintf('\nAllocation Uniformity: Min=%d, Max=%d, Mean=%.2f, Std=%.2f\n', ...
                min(valid_allocations), max(valid_allocations), ...
                mean(valid_allocations), std(valid_allocations));
    end
end