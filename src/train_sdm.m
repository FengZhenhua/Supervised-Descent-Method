%
% Supervised descent method for facial landmark detection and face tracking
%
% Subfunction used for training of cascaded linear regressors in SDM
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

function sdm_model = train_sdm(train_img, train_init_shape, train_gt_shape, cr_model)

num_train_samples = length(train_img);

%% initialise the current shape
flag = 1;
for j = 1:num_train_samples
    for i = 1 : size(train_init_shape(1).data,2)
        current_shape(:,flag) = train_init_shape(j).data(:,i);
        flag = flag + 1;
    end
end
error_shape = zeros(size(current_shape));

%% train cascaded linear regressors in sdm
for step = 1:cr_model.deepth
    disp(['Training the ' num2str(step) '-th weak regressor ...']);
    
    tic;
    
    % obtain shape-indexed local features
    feature_matrix = [];
    flag = 1;
    for i = 1:num_train_samples
        for j = 1:size(train_init_shape(i).data,2)
            error_shape(:,flag) = train_gt_shape(:, i) - current_shape(:,flag);
            feature_matrix(:,flag)  = obtain_features(train_img(i).data, current_shape(:,flag), cr_model);
            flag = flag + 1;
        end
    end
    feature_matrix = [feature_matrix; ones(1,size(feature_matrix,2))];
    
    tmp = toc;
    disp(['Obtaining shape-indexed local features costs ', num2str(tmp), 's']);
    disp(['Shape-indexed local features demension: ', num2str(size(feature_matrix,1))]);
    
    pennlty = eye(size(feature_matrix,1)) * cr_model.lambda;
    pennlty(end,end) = 0;
    
    tic;
    
    sdm_model(step).A = (feature_matrix  *  feature_matrix' + pennlty) \ feature_matrix * error_shape';
    
    tmp = toc;
    disp(['Caculating regressors costs ', num2str(tmp), 's']);
    
    current_shape = current_shape + sdm_model(step).A(1:end-1,:)' * feature_matrix(1:end-1,:) + repmat(sdm_model(step).A(end,:)', 1, size(current_shape,2));
    disp(['Training the ' num2str(step) '-th weak regressor finish ...']);
end
end

