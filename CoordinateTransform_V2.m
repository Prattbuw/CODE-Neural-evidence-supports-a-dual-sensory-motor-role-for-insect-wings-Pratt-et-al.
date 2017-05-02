function[RotatedWing, X_Rotated, Y_Rotated, Z_Rotated]=CoordinateTransform_V2(filename,WTip, WBlade,N)
% written by Tanvi Deora, August 2016 to transform the cartesian space from the
% calibration object space to the wing specfic space. Inputs are 1)
% digitized points on the wings as a .csv file. Input that filename with 'filename.csv' 2) digitized point number that corresponds to the wing tip 3)point
% number for a point on the wing blade (xy plane) that is towards the y direction if the wing base to wing tip is the x direction 
%N=Numframes
%% get you variables
%name='filename';
Digitized_pts=csvread(filename,1 ,0);

%% Translate the points to Center the wing

Wing_Avg=mean(Digitized_pts,1);
Length_pts=length(Wing_Avg);
Avg_X=mean(Wing_Avg(1,1:3:end-2)).';
Avg_Y=mean(Wing_Avg(1,2:3:end-1)).';
Avg_Z=mean(Wing_Avg(1,3:3:end)).';
Avg_vec=[Avg_X,Avg_Y,Avg_Z];

% The next line was used when checking this code for accuracy with a calibration object: it is the point used as the orgin for the calibration
% Avg_vec=mean([Digitized_pts(:,1),Digitized_pts(:,2), Digitized_pts(:,3)-5.2]); 
% Length_pts=length(Digitized_pts(1,:));

% so the Avg_vec is the true center point of the wing calculated across all
% the frames.
Repeat_Avg_vec=repmat(Avg_vec,[N,(Length_pts./3)]);

% center each point on the wing
Wing_points_Centered=Digitized_pts-Repeat_Avg_vec;
%% Rotate matrix along two axes to define the new wing centric cartesian system
% Define the wing tip as the x axis and another point on the wing blade to
% define the wing surface

% Use the average coordinate of the wing tip point(WTip) - the wing tip point to create what would be a
% potential x axis in the wing aligned coordinate system
Wing_Tip_Point=mean(Wing_points_Centered(:,WTip*3-2:WTip*3));

% use the average coorrdinate for another point, wingBlade point (WBlade)to define the wing
% blade along with the Wing tip point
Wing_Blade_Point=mean(Wing_points_Centered(:,WBlade*3-2:WBlade*3));

% finding the normal to the wing blade
Cross_Product_XBladePoint=cross(Wing_Tip_Point, Wing_Blade_Point);
UnitZdash = Cross_Product_XBladePoint./norm(Cross_Product_XBladePoint);

% find the angle between the two z axis to rotate along the y axis
Cartisan_Z=[0,0,1];
%Phase angle method 1
%Plane_Angle=(atan(norm(cross(Cross_Z,Cartisan_Z),2)./(dot(Cross_Z,Cartisan_Z)))).*(180./pi);

%Phase angle method 2 using cos
DotProduct = dot(UnitZdash,Cartisan_Z); % actually n1.n2 = |n1||n2|cosang
Angle_z = 1.*(acos((DotProduct./(norm(UnitZdash).*norm(Cartisan_Z)))));
%rotate
%RotateY is the rotation matrix along y
RotateY=[cos(Angle_z), 0, sin(Angle_z); 0, 1, 0; -sin(Angle_z), 0. cos(Angle_z)];
Wing_rotatedY=NaN*zeros(N,Length_pts);
for i=1:N
    for j=1:3:Length_pts-2
   Wing_rotatedY(i,j:j+2)=RotateY*[Wing_points_Centered(i,j), Wing_points_Centered(i,j+1),Wing_points_Centered(i,j+2)]';
     end
end

%New Unit_x vector
New_Wing_Tip_Point=mean(Wing_rotatedY(:,WTip*3-2:WTip*3));
New_UnitXdash = New_Wing_Tip_Point./norm(New_Wing_Tip_Point);
%Rotate along z
%Find angle between x axis
clear i;
clear j;
Cartisan_X=[1,0,0];
Angle_x = 1.*(acos((dot(New_UnitXdash, Cartisan_X))/(norm(New_UnitXdash)*norm(Cartisan_X))));
RotateZ=[cos(Angle_x) -sin(Angle_x) 0; sin(Angle_x) cos(Angle_x) 0; 0 0 1];
Wing_rotatedZ=NaN*zeros(N,Length_pts);
for i=1:N
    for j=1:3:Length_pts-2
   Wing_rotatedZ(i,j:j+2)=RotateZ*[Wing_rotatedY(i,j), Wing_rotatedY(i,j+1),Wing_rotatedY(i,j+2)]';
     end
end

RotatedWing=Wing_rotatedZ;
X_Rotated=RotatedWing(:,1:3:end-2);
Y_Rotated=RotatedWing(:,2:3:end-1);
Z_Rotated=RotatedWing(:,3:3:end);