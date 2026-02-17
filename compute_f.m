function [f,p,r] = compute_f(T,H)
    % Compute F-measure, precision, and recall
    
    if length(T) ~= length(H)
        error('The length of T and H must be equal.');
    end
    
    N = length(T);
    numT = 0;
    numH = 0;
    numI = 0;
    
    for n = 1:N-1
        Tn = (T(n+1:end)) == T(n);
        Hn = (H(n+1:end)) == H(n);
        
        Tn = Tn(:);
        Hn = Hn(:);
        
        numT = numT + sum(Tn);
        numH = numH + sum(Hn);
        numI = numI + sum(Tn .* Hn);
    end
    
    p = 1;
    r = 1;
    f = 1;
    
    if numH > 0
        p = numI / numH;
    end
    
    if numT > 0
        r = numI / numT;
    end
    
    if (p + r) > 0
        f = 2 * p * r / (p + r);
    else
        f = 0;
    end
end