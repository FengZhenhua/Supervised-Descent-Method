%
% Supervised descent method for facial landmark detection and face tracking 
%
% Subfunction used for projecting a shape to a face bounding box
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

function Out_Shape = project_s2b(In_Shape, Bbox)
In_Shape = reshape(In_Shape,[],2);
Out_Shape = In_Shape;
Out_Shape(:,1) = Out_Shape(:,1) / (max(Out_Shape(:,1)) - min(Out_Shape(:,1))) * Bbox(3);
Out_Shape(:,2) = Out_Shape(:,2) / (max(Out_Shape(:,2)) - min(Out_Shape(:,2))) * Bbox(4);
Out_Shape(:,1) = Out_Shape(:,1) - min(Out_Shape(:,1)) + Bbox(1);
Out_Shape(:,2) = Out_Shape(:,2) - min(Out_Shape(:,2)) + Bbox(2);
Out_Shape = Out_Shape(:);
end

