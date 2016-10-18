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

DEBUG = 0; % set this value to 1 to show intermediate results

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

%% generate initial shapes for each training image
% !!!!! It should be noted that you can generate multiple initial shapes for each face image !!!!! 
% !!!!! This will greatly improve the performance of your trained model  !!!!!

mean_shape = mean(train_gt_shape,2);
for i = 1:length(train_gt_shape)
    for j = 1
        train_init_shape(i).data(:,j) = project_s2b(mean_shape, train_bbox(:,i));
    end
end

%% debug
if DEBUG
    for i = 1:10
        imshow(train_img(i).data);
        hold on;
        plot(train_init_shape(i).data(1:end/2,1), train_init_shape(i).data(end/2+1:end,1), 'y*');
        plot(train_gt_shape(1:end/2,i), train_gt_shape(end/2+1:end,i), 'ro');
        hold off;
        pause(0.5);
    end
end

%% train cascaded linear regressors in SDM
% intialise model parameters, you can tune these parameters using cross validation
cr_model.deepth = 5; %number of weak regressors in cascade
cr_model.lambda = 1000; %weight of the regularisation term
cr_model.patch_radius = 16; %radius of the local patch around a landmark, used for extracting local features
cr_model.mean_shape = mean_shape; %mean shape
cr_model.mean_face_size = mean_face_size; %mean face size

% train a new model 
% !!! or load a pre-trained model from our data by uncommting the next line
% load('./model/cr_model_cofw.mat');
cr_model.model = train_sdm(train_img, train_init_shape, train_gt_shape, cr_model);
save('./model/cr_model_cofw.mat', 'cr_model');

%% apply the trained model to test images
for i = 1:length(test_img)
    face_size = test_bbox(3,i) * test_bbox(4,i);
    scale = cr_model.mean_face_size / face_size;
    tmp_img = imresize(test_img(i).data, scale);
    tmp_bbox = test_bbox(:,i) * scale;
    init_shape = project_s2b(cr_model.mean_shape, tmp_bbox);
    predict_shape(:,i) = fit_sdm(tmp_img, init_shape, cr_model) / scale;
    
    if DEBUG
        imshow(test_img(i).data);
        hold on;
        plot(predict_shape(1:end/2,i), predict_shape(end/2+1:end,i), 'y*');
        plot(test_gt_shape(1:end/2,i), test_gt_shape(end/2+1:end,i), 'ro');
        hold off;
        pause(0.5);
    end
end

%% plot results
% caculating the normalised fitting error for each test image
for i = 1:length(test_img)
    error(i) = 0;
    for j = 1:size(predict_shape,1)/2
        error(i) = error(i) + sqrt((predict_shape(j,i) - test_gt_shape(j,i))^2 + (predict_shape(j+end/2,i) - test_gt_shape(j+end/2,i))^2);
    end
    eye_distance_vector = [test_gt_shape(17,i) - test_gt_shape(18,i), test_gt_shape(17+end/2,i) - test_gt_shape(18+end/2,i)];
    error(i) = error(i) / norm(eye_distance_vector)/(size(predict_shape,1)/2);
end

disp(['Average error normalised by inter-ocular distance: ' num2str(sum(error)/length(error))]);

% plot the cumulative error distribution curve
error = sort(error);
plot(error, [1:length(error)]/length(error), 'm-', 'linewidth', 2);
xlim([0 0.2]);
ylim([0 1]);
grid on;
legend('SDM');
xlabel('Normalised error by inter-ocular distance');
ylabel('Propotion of test images');
title('cumulative error distribution curve')


