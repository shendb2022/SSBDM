function [L] = calculateLgByRBF(X, sigma)
[n, ~] = size(X);
A = zeros(n, n);
for i = 1:n
    xi = X(i,:);
    for j = i+1:n
        xj = X(j,:);
        distance = sum((xi-xj).^2);
%         cos_sim = (xi * xj') / (norm(xi,2)*norm(xj,2) + eps);
%         cos_sim = min(1, max(-1, cos_sim));   % 保证在 [-1, 1] 之间
%         sad = acos(cos_sim);
%         sad = exp(-1 * sad / 0.0001);
%         A(i, j) = sad;
        A(i, j) = exp(-1 * distance / sigma ^ 2);
        A(j, i) = A(i, j);
    end
end
D = sum(A, 2);
D = diag(D);
L = D - A;
end