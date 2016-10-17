%
% Supervised descent method for facial landmark detection and face tracking
%
% Subdunction used for extracting shape-indexed local features
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

function features = obtain_features(img, shape, cr_model)
features = [];
for i=1:size(shape,1)/2
    x = floor(shape(i));
    y = floor(shape(i+end/2));
    IX_X = x - cr_model.patch_radius + 1 : x + cr_model.patch_radius;
    IX_Y = y - cr_model.patch_radius + 1 : y + cr_model.patch_radius;
    IX_Y(IX_Y<1) = 1;
    IX_X(IX_X<1) = 1;
    IX_Y(IX_Y>size(img,1)) = size(img,1);
    IX_X(IX_X>size(img,2)) = size(img,2);
    tmp = img(round(IX_Y), round(IX_X),:);
    hogf = extractHOGFeatures(tmp);
    features = [features; hogf(:)];
end
end