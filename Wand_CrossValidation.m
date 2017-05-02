%% Video Calibration Wand Cross Validation
% Created by Brandon Pratt
% Date: October 2016
clear all; close all; clc;
%% Load Data
filename='Digitized_Wandxyzpts.csv';
Digitized_pts=csvread(filename,1 ,0);
%% Isolate Point XYZ
% Measured wand length
Measured_Wand_Length=70.19; %Average of measurements 70.21,70.17,& 70.18

% Digitized pt coordinates
% Point 1
Pt1_X=Digitized_pts(:,1);
Pt1_Y=Digitized_pts(:,2);
Pt1_Z=Digitized_pts(:,3);

% Point 2
Pt2_X=Digitized_pts(:,4);
Pt2_Y=Digitized_pts(:,5);
Pt2_Z=Digitized_pts(:,6);

%% Find the Distance between points at each frame
% Change in xyz
delta_X=Pt1_X-Pt2_X;
delta_Y=Pt1_Y-Pt2_Y;
delta_Z=Pt1_Z-Pt2_Z;

% Calculate distance vector (Digitized Wand Length) at each frame
Digitized_Wand_Length=sqrt(delta_X.^2 + delta_Y.^2 + delta_Z.^2);
Rep_Measured_Wand_Length=repmat(Measured_Wand_Length,length(Digitized_Wand_Length),1);
Mean_Digitized_Wand_Length=repmat(mean(Digitized_Wand_Length),length(Digitized_Wand_Length),1);
%% Plot Digitized and Measured Wand Length
plot(Digitized_Wand_Length,'ro');
hold on;
plot(Rep_Measured_Wand_Length,'k','LineWidth',3)
plot(Mean_Digitized_Wand_Length,'b','LineWidth',2)
xlabel('Frame Number(#)')
ylabel('Wand Length(mm)')
legend('Digitized Wannd Length','Measured Wand Length','Mean Digitized Wand Length')
%% Calculate Error (Root mean squared error)
Wand_Length_Error=Rep_Measured_Wand_Length-Digitized_Wand_Length;
Sq_Wand_Length_Error=Wand_Length_Error.^2;
Mean_Sq_Wand_Length_Error=mean(Sq_Wand_Length_Error);
Sqrt_Mean_Sq_Wand_Length_Error=sqrt(Mean_Sq_Wand_Length_Error);
