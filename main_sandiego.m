clear all;  clc; close all
addpath("dictionary");
data_name = "Sandiego";  % For better performance, you can tune the parameters alpha and beta in Line 89 and 90
load(data_name);
% mex -setup cpp
% mex hybridseg.cpp
% maxNumCompThreads(1);
% rng(0);
tic;
data = double(data);
d = double(d);
[nrows,ncols,nbands] = size(data);
%% Normalization using the max-min style
data = hyperNormalize(data);
data_matrix = reshape(data, nrows * ncols, nbands);
X = data_matrix';

% load background dictionary
load(cell2mat([data_name, '_bg_dictionary.mat']));
A = double(A);
% PCA is applied to obtain the main information for superpixel segmentation
n_components = 10;
pca_data = fPCA_2D_SpecificV1(data_matrix,1,1,0);
pca_data = pca_data(:,1:n_components);
HIM = reshape(pca_data,nrows,ncols,n_components);
clear pca_data;

% reshape the data into a vector
input_img = zeros(1, nrows * ncols * n_components);

startpos = 1;
for i = 1 : nrows % lines
    for j = 1 : ncols % columes
        input_img(startpos : startpos + n_components - 1) = HIM(i, j, :); % bands
        startpos = startpos + n_components;
    end
end
s_ratio = 0.010;
numSuperpixels = floor(nrows * ncols * s_ratio); % the desired number of superpixels
compactness = 0.01; % compactness2 = 1-compactness, the clustering is according to: compactness*dxy+compactness2*dspectral
dist_type = 2; % distance type - 1:Euclidean；2：SAD; 3:SID; 4:SAD-SID
seg_all = 1; % 1: all the pixles participate in the clustering， 2: some pixels would not
% labels: segmentation results
% numlabels: the final number of superpixels
% seedx: the x indexing of seeds
% seedy: the y indexing of seeds
[labels, numlabels, seedx, seedy] = hybridseg(input_img, nrows, ncols, n_components, numSuperpixels, compactness, dist_type, seg_all);%numlabels is the same as number of superpixels
clear input_img;
segment_label = zeros(nrows, ncols);
startpos = 1;
for i = 1 : nrows % lines
    for j = 1 : ncols % columes
        segment_label(i, j, :) = labels(startpos : startpos);
        startpos = startpos + 1;
    end
end
% % calculate the size of every superpixel
max_size = 0;
opt_i = 0;
numSuperpixels = max(segment_label(:));
nums = zeros(1,numSuperpixels);
for i = 1:numSuperpixels
    i_size = size(find(segment_label == i),1);
    nums(i)= i_size;
    if i_size > max_size
        max_size = i_size;
        opt_i = i;
    end
end
%calculate the graph Laplacian matrix
Lgs = zeros(numSuperpixels, max_size, max_size);
t_labels = reshape(segment_label, nrows * ncols, 1);
for i = 1:numSuperpixels
    [cols, rows] = find(segment_label' == i);%matlab column first while python row first
    num = size(cols, 1);
    super_pixel = find_pixels(data, num, rows, cols, nbands);
    Lgs(i, 1:num,1:num) = calculateLgByRBF(super_pixel, 0.5);
end
% save('Lgs.mat','Lgs');
% load segment_label
% load Lgs
numSuperpixels = max(segment_label(:));
% solve the SSADM by updating the endmembers and abundances iternatively
max_auc = 0;
opt_alpha = 0;
opt_beta = 0;
% for alpha = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3]
%     for beta = [0 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 1e0 1e1 1e2 1e3]
alpha = 1e-2; % 1e-3
beta = 1e-1; % 0.1
mu = 1e-1;
iter = 1e2;
episilon = 1e-3;
B = [A, d];
Ads = A;
for i = 1:size(B, 2)-1
    for j = i+1:size(B, 2)
        Ads = [Ads, B(:,i).*B(:,j)];
    end
end
M = [Ads, d];
% M = B;
% T = SSADM3_no_anc_mu_fixed_weighted3(X, M, segment_label, Lgs, numSuperpixels, mu, alpha, beta, iter, episilon);
T = SSBDM3_noweighted_sparsity(X, M, segment_label, Lgs, numSuperpixels, mu, alpha, beta, iter, episilon);
detection_map = reshape(sqrt(sum((T).^2)), nrows, ncols);
detection_map = hyperNormalize(detection_map);
toc;
map = detection_map;
map = exp(-1 * (map - 1).^2 / 0.16);
% save('../SSBDM_Results/Sandiego/SSBDM.mat','map');
imagesc(map);axis off
[TPR, FPR,tau] = roc(groundtruth(:)',detection_map(:)');
AUC = trapz(FPR,TPR);
[tau_sorted, idx] = sort(tau, 'ascend');
FPR_sorted = FPR(idx);
TPR_sorted = TPR(idx);
AUC_back = trapz(tau_sorted, FPR_sorted);
AUC_tar = trapz(tau_sorted, TPR_sorted);
AUC_oa = AUC + AUC_tar - AUC_back;
disp(AUC);
disp(AUC_oa);
