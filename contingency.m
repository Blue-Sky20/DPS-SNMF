function Cont = contingency(Mem1, Mem2)
% Form contingency matrix for two cluster assignment vectors

if nargin < 2 || min(size(Mem1)) > 1 || min(size(Mem2)) > 1
   error('Contingency: Requires two vector arguments');
end

Mem1 = Mem1(:);
Mem2 = Mem2(:);

if length(Mem1) ~= length(Mem2)
    error('Contingency: Input vectors must have the same length');
end

% Handle invalid labels
Mem1 = fix_labels(Mem1);
Mem2 = fix_labels(Mem2);

[~, ~, Mem1] = unique(Mem1);
[~, ~, Mem2] = unique(Mem2);

n1 = max(Mem1);
n2 = max(Mem2);
Cont = zeros(n1, n2);

for i = 1:length(Mem1)
    if Mem1(i) > 0 && Mem2(i) > 0 
        Cont(Mem1(i), Mem2(i)) = Cont(Mem1(i), Mem2(i)) + 1;
    end
end
end

function labels = fix_labels(labels)
% Subfunction to fix invalid labels
labels(isnan(labels)) = 0;
labels(isinf(labels)) = 0;
labels(labels < 0) = 0;
labels = round(labels);
end