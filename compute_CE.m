function CE = compute_CE(A, A0)
% Compute the clustering error between predicted and true assignments

A = double(A(:)');
A0 = double(A0(:)');

L = max(A);
L0 = max(A0);

N = numel(A);

% Map to consecutive labels
Label1 = unique(A0);
for i = 1:length(Label1)
    A0(A0 == Label1(i)) = i;
end

Label2 = unique(A);
for i = 1:length(Label2)
    A(A == Label2(i)) = i;
end

% Calculate error matrix
Aerror = zeros(L, max(L, L0));
for i = 1:L
    for j = 1:L0
        Aerror(i, j) = nnz(A0(A == i) ~= j);
    end
    Aerror(i, L0+1:end) = nnz(A == i);
end

% Find optimal mapping
if max(L, L0) <= 10
    perm = double(perms(1:max(L, L0)));
    perm = perm(:, 1:L);
    
    base = repmat((0:L-1)*L, size(perm, 1), 1);
    ind_set = base + perm; 
    
    [CE, ~] = min(sum(Aerror(ind_set), 2));
    CE = CE / N;
else
    swap = [];
    for i = 2:L
        swap = [swap; repmat(i, i-1, 1) (1:i-1)'];
    end
    
    CE = N;
    for i = 1:10
        perm = randperm(L);
        
        for j = 1:1e5
            idx1 = sub2ind(size(Aerror), swap(:,1), perm(swap(:,2))');
            idx2 = sub2ind(size(Aerror), swap(:,2), perm(swap(:,1))');
            idx3 = sub2ind(size(Aerror), swap(:,1), perm(swap(:,1))');
            idx4 = sub2ind(size(Aerror), swap(:,2), perm(swap(:,2))');

            diff_vals = Aerror(idx1) + Aerror(idx2) - Aerror(idx3) - Aerror(idx4);
            [m, ind] = min(diff_vals);
            if m >= 0, break; end
            
            temp = perm(swap(ind,1));
            perm(swap(ind,1)) = perm(swap(ind,2));
            perm(swap(ind,2)) = temp;
        end
        
        if sum(Aerror(sub2ind(size(Aerror), 1:L, perm))) < CE
            CE = sum(Aerror(sub2ind(size(Aerror), 1:L, perm)));
        end
    end
    
    CE = CE / N;
end
end