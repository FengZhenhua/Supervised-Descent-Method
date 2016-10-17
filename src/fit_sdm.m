%
% Supervised descent method for facial landmark detection and face tracking
%
% Subdunction used for fitting SDM to a new image
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

function predict_shape = fit_sdm(img, init_shape, cr_model)

predict_shape = init_shape;

for i = 1 : cr_model.deepth
    feature = obtain_features(img, predict_shape, cr_model);
    predict_shape = predict_shape + cr_model.model(i).A(1:end-1,:)' * feature + cr_model.model(i).A(end,:)';
end

predict_shape = predict_shape(:);
end