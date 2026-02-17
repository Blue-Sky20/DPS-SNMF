function [W, D] = ConstructKNN_Graph(X, k)
    % Construct a k-nearest neighbors graph from multi-view data
    nviews = length(X);
    n = size(X{1}, 2);
    dist_sum = zeros(n, n);

    for v = 1:nviews
        Data = X{v}';
        Data = Data ./ (sqrt(sum(Data.^2, 2)) + 1e-10);
        S = sum(Data.^2, 2);
        dist = bsxfun(@plus, S, S') - 2 * (Data * Data');
        dist_sum = dist_sum + dist;
    end

    W = zeros(n, n);
    [~, idx] = sort(dist_sum, 2, 'ascend');

    for i = 1:n
        id = idx(i, 2:k+1);
        W(i, id) = 1;
    end

    W = (W + W') / 2;
    D = diag(sum(W, 2));
end