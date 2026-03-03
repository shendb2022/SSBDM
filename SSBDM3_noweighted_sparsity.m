function tZ = SSADM3_noweighted_sparsity(X, M, segment_label, Lgs, super_num, mu0, alpha, beta, iter, episilon)
% Estimate the abundances and background endmembers using SSADM and physical constraints 
% model is min_{Y}1/2\|X-MY\|_F^2+alpha*Sigma_g=1^{Ng}Tr(YgLgYg')+*beta\|W2W1*Y\|_{1, 1}
% Usage 
%   SSADM(X, M, segment_label, Lgs,super_num, alpha, beta, gamma)
% Inputs
%   X - Input image (L*N)
%   M - Spectral dictionary (L*Q)
%   segment_label - Map of superpixels (h * w)
%   Lgs - Laplacian matrix of all superpixels (super_num * max_size * max_size)
%   m - Number of endmembers
%   super_num - Total number of superpixels
%   alpha, beta, gamma - Three regularization parameters
% Outputs
%   T - Target components, i.e., T = t * S(m,:)(L*N)
% loss_list = [];
Q = size(M, 2);
[L, N] = size(X);
[h, w] = size(segment_label);

% Variables to be optimized in the subproblem of abundance estimation
V1 = zeros(L, N);
V2 = zeros(Q, N);
V3 = zeros(Q, N);

% Lagrange multipliers to connect the major variables and the auxiliary
% variables
D1 = zeros(L, N);
D2 = zeros(Q, N);
D3 = zeros(Q, N);

% The epision for the covergence tolerance 
epsilon = episilon;

% The maximum iteration
maxIter = iter;

mu = mu0;
rho = 1.1;
mu_max = 1e30;

% To avoid repeated calculations
MTM = M' * M;
pre_comp = MTM + 2 * eye(Q);
% normX = norm(X(:));

% Small constant
cons = 1e-8;

% Iteratively update the abundances and endmembers
for iter = 1:maxIter
    % Update Y
    right = M' * (V1 + D1) + V2 + D2 + V3 + D3;
    Y = pre_comp \ right;
    MY = M * Y;
    % Update V1
    V1 = (X + mu * (MY - D1)) / (1+mu);
    % Update V2
    Y_tensor = reshape(Y', h, w, Q);
    V2_tensor = reshape(V2', h, w, Q);
    D2_tensor = reshape(D2', h, w, Q);
    for g = 1:super_num
        [cols, rows] = find(segment_label' == g);%matlab column first while python row first
        n_pixels = size(cols, 1);
        Yg = find_pixels(Y_tensor, n_pixels, rows, cols, Q);
        D2g = find_pixels(D2_tensor, n_pixels, rows, cols, Q);
        left = mu * (Yg' - D2g');
        right = 2 * alpha * squeeze(Lgs(g, 1:n_pixels, 1:n_pixels)) + mu * eye(n_pixels);
        V2g = (left / right)';
        V2_tensor = inject_pixels(V2_tensor, V2g, n_pixels, rows, cols);
    end
    V2 = reshape(V2_tensor, h * w, Q);
    V2 = V2';
    % Update V3
    temp = Y - D3;
%     W1 = 1 ./ (temp + cons);
%     W11 = sum(B.^2,1)' + sum(BS.^2,1) - 2 * (B' * BS);
%     W11 = 1 ./ (1 + exp(-0.5 * W1));
    % 正则化列向量（L2归一化）
    A_norm = MY ./ (vecnorm(MY)+cons);  % L x N
    B_norm = M ./ (vecnorm(M)+cons);  % L x m

    % 计算余弦相似度（B 的每列和 A 的每列做内积）
    cos_sim = B_norm' * A_norm;  % m x N
% 
%     % 转换为余弦距离
    W12 = 1 - cos_sim;  % m x N
    W1 = W12;
%     disp(size(W1));
    W2 = vecnorm(temp, 2, 2);
    W2 = diag(1 ./ (W2 + cons));
%     V3 = soft(temp, beta / mu);
%     V3 = soft(temp, beta / mu.*W1);
    V3 = soft(temp, beta / mu.*W2*W1);
    % Upadte H1, H2
    diff_V1 = MY - V1;
    diff_V2 = Y - V2;
    diff_V3 = Y - V3; 
    D1 = D1 - diff_V1;
    D2 = D2 - diff_V2;
    D3 = D3 - diff_V3;
    % Result T
    t = M(:, Q:Q);
    Z = Y(Q:Q, :);
    tZ = t * Z;
    % Update mu
    mu = min(mu * rho, mu_max);
    % Calculate the tolerance
    stop_diff = norm(diff_V1(:)) + norm(diff_V2(:))+norm(diff_V3(:)) ;
%     mu = min(mu * rho, mu_max);
%     loss_list = [loss_list, stop_diff];
    if iter==1 || mod(iter,50)==0 || stop_diff<epsilon
        disp(['iter ' num2str(iter) ',mu=' num2str(mu,'%2.1e') ...
            ',stopALM=' num2str(stop_diff,'%2.3e')]);
    end
    if stop_diff<epsilon
        break;
    end
end
% save('loss_list_abu-airport-4');




