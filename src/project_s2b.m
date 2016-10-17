% fit a face shape to a bbox
function Out_Shape = project_shape_to_bbox(In_Shape, Bbox)
In_Shape = reshape(In_Shape,[],2);
Out_Shape = In_Shape;
Out_Shape(:,1) = Out_Shape(:,1) / (max(Out_Shape(:,1)) - min(Out_Shape(:,1))) * Bbox(3);
Out_Shape(:,2) = Out_Shape(:,2) / (max(Out_Shape(:,2)) - min(Out_Shape(:,2))) * Bbox(4);
Out_Shape(:,1) = Out_Shape(:,1) - min(Out_Shape(:,1)) + Bbox(1);
Out_Shape(:,2) = Out_Shape(:,2) - min(Out_Shape(:,2)) + Bbox(2);
Out_Shape = Out_Shape(:);