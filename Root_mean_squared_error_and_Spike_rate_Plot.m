%% Root mean squared error (RMSE) Vs Spike Rate 
% NOTE: Relative Error in the document refers to root mean squared error
% Created by: Brandon Pratt
% Date: Feb. 2017
clear all; close all; clc
%% Add Paths and Load Data
addpath('NonBase Unit Encoding Results')
addpath('Base Unit Encoding Properties Results')

% Load Average recrodered spike rate and root mean square error error during three
% ampltitudes of a sinusoidal stimulus for base and non-localized units
load('Non-localized Quantitative Spike Rate Vectors.mat')
% Non-localized units
% NOTE: ONLY COMPARE THOSE UNITS THAT HAD A SPIKE RATE GREATER THAN ZERO
% THROUGHT DESIRED TIME WINDOW
% Average Spike Rate
AVE_Actual_SR_Store=AVE_Actual_SR_Store([1:24,26:end],:); %orginal plot: ~=0
Non_Localized_Ave_SR_Amp1=AVE_Actual_SR_Store(find(AVE_Actual_SR_Store(:,1)~=0),1);
Non_Localized_Ave_SR_Amp2=AVE_Actual_SR_Store(find(AVE_Actual_SR_Store(:,2)~=0),2);
Non_Localized_Ave_SR_Amp3=AVE_Actual_SR_Store(find(AVE_Actual_SR_Store(:,3)~=0),3);

% Relative Error (RMSE)
Rel_Error_Store=Rel_Error_Store([1:24,26:end],:);
Non_Localized_Rel_Err_Amp1=Rel_Error_Store(find(AVE_Actual_SR_Store(:,1)~=0),1);
Non_Localized_Rel_Err_Amp2=Rel_Error_Store(find(AVE_Actual_SR_Store(:,2)~=0),2);
Non_Localized_Rel_Err_Amp3=Rel_Error_Store(find(AVE_Actual_SR_Store(:,3)~=0),3);

% Clear Variables
clearvars AVE_Actual_SR_Store Rel_Error_Store

% Base Localized Units
load('Base Quantitative Spike Rate Vectors.mat');
% NOTE: ONLY COMPARE THOSE UNITS THAT HAD A SPIKE RATE GREATER THAN ZERO
% THROUGHT DESIRED TIME WINDOW
% Average Spike Rate
Base_Localized_Ave_SR_Amp1=AVE_Actual_SR_Store(find(AVE_Actual_SR_Store(:,1)~=0),1);
Base_Localized_Ave_SR_Amp2=AVE_Actual_SR_Store(find(AVE_Actual_SR_Store(:,2)~=0),2);
Base_Localized_Ave_SR_Amp3=AVE_Actual_SR_Store(find(AVE_Actual_SR_Store(:,3)~=0),3);

% Relative Error (RMSE)
Base_Localized_Rel_Err_Amp1=Rel_Error_Store(find(AVE_Actual_SR_Store(:,1)~=0),1);
Base_Localized_Rel_Err_Amp2=Rel_Error_Store(find(AVE_Actual_SR_Store(:,2)~=0),2);
Base_Localized_Rel_Err_Amp3=Rel_Error_Store(find(AVE_Actual_SR_Store(:,3)~=0),3);

%% Root mean squared error Vs. Average Spike Rate Plot

% Plot Non-localized Units
% Amplitude 1= 4.4 mm total displacement
plot(Non_Localized_Ave_SR_Amp1,Non_Localized_Rel_Err_Amp1,'ko','MarkerSize',8)
hold on; 

% Amplitude 2= 8.8 mm total displacement
plot(Non_Localized_Ave_SR_Amp2,Non_Localized_Rel_Err_Amp2,'ro','MarkerSize',8)

% Amplitude 3= 13.2 mm total displacement
plot(Non_Localized_Ave_SR_Amp3,Non_Localized_Rel_Err_Amp3,'bo','MarkerSize',8)

% Plot Base-localized Units
% Amplitude 1= 4.4 mm total displacement
plot(Base_Localized_Ave_SR_Amp1,Base_Localized_Rel_Err_Amp1,'k*','MarkerSize',8)

% Amplitude 2= 8.8 mm total displacement
plot(Base_Localized_Ave_SR_Amp2,Base_Localized_Rel_Err_Amp2,'r*','MarkerSize',8)

% Amplitude 3= 13.2 mm total displacement
plot(Base_Localized_Ave_SR_Amp3,Base_Localized_Rel_Err_Amp3,'b*','MarkerSize',8)

% Plot Settings
%title('Spike Rate less than 1','FontSize',12)
ylabel('RMSE','FontSize',12)
xlabel('Average Spike Rate (Spikes/Sec)','FontSize',12)
h=gca;
set(h,'FontSize',12)
legend('Non-Localized Amplitude 1','Non-Localized Amplitude 2','Non-Localized Amplitude 3',...
    'Base-Localized Amplitude 1','Base-Localized Amplitude 2','Base-Localized Amplitude 3');

