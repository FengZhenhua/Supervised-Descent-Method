%
% Supervised descent method for facial landmark detection and face tracking 
%
% Example code for facial landmark detection
%
% Copyright @ Zhenhua Feng, fengzhenhua2010@gmail.com
% Centre for Vision, Speech and Signal Processing, University of Surrey
%
% Please cite the following papers if you are using this code
%
% 1. Feng, Z. H. et al. Random cascaded-regression copse for robust facial landmark detection. IEEE SPL, 2015, 1(22), pp:76-80.
% 2. Feng, Z. H. et al. Cascaded collaborative regression for robust facial landmark detection trained using a mixture of synthetic and real images with dynamic weighting. IEEE TIP, 2015, 24(11), pp:3425-3440.
% 3. Xiong, X., & De la Torre, F. Supervised descent method and its applications to face alignment. CVPR, 2013, pp:532-539.
%

close all;
clear;
clc;

addpath('./src/');

%% load training data and test data of the COFW dataset, inlcuding face images, landmarks and face bounding boxes
load('./data/COFW_train_color.mat');
load('./data/COFW_test_color.mat');

% convert the data format of COFW to ours
for i = 1:length(IsTr)
    if size(IsTr{i},3) == 3
        train_img(i).data = rgb2gray(IsTr{i});
    else
        train_img(i).data = IsTr{i};
    end
end
train_gt_shape = phisTr(:,1:58)';
train_bbox = bboxesTr';

for i = 1:length(IsT)
    if size(IsT{i},3) == 3
        test_img(i).data = rgb2gray(IsT{i});
    else
        test_img(i).data = IsT{i};
    end
end
test_gt_shape = phisT(:,1:58)';
test_bbox = bboxesT';

clear bboxesT bboxesTr IsT IsTr phisT phisTr;

%% resize all the training faces 
mean_face_size = sum(train_bbox(3,:) .* train_bbox(4,:))/length(train_bbox); 
for i = 1:length(train_img)
    face_size = train_bbox(3,i) * train_bbox(4,i);
    scale = mean_face_size / face_size;
    train_img(i).data = imresize(train_img(i).data, scale);
    train_gt_shape(:,i) = train_gt_shape(:,i) * scale;
    train_bbox(:,i) = train_bbox(:,i) * scale;
end

%% generate initial shape for each training image
mean_shape = mean(train_gt_shape,2);
for i = 1:length(train_gt_shape)
    train_init_shape(:,i) = project_s2b(train_gt_shape(:,i), train_bbox(:,i));
end

%% train cascaded linear regressors in SDM
sdm_para.deepth = 5;
sdm_para.feature = 'hog';
sdm_para.patch_size = 32;
sdm_para.cell_size = 10;

cr_model = train_sdm(train_img, train_init_shape, train_gt_shape, sdm_parameters);
% or load a pre-trained model from our data 
load('./model/cr_model_cofw.mat');

%% apply the trained model to test images
for i = 1:length(test_img)
    
end

%% plot results




