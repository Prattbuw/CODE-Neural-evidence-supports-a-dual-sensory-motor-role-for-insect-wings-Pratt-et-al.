%% Build Vectors for Spike Prediction Analysis
% Created by: Brandon Pratt
% Date: January 2017
clear all; close all; clc

%% Load Data
% Load Spike Trains and Identity of Units that are Base Idenified
load('Wing_Base_Identity_Spike_Trains.mat')
% Results path for base units
ResultPath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code/Base Unit Encoding Properties Results/';
Coh_Base_Units=[3,4,5,9,10,11,12,14,19,20,23,27,28,31];

% NonBase Spike Trains
% Load Spike Trains and Identity of Units that are Base Idenified
%load('Wing_NonBase_Identity_Spike_Trains.mat')
%Coh_Base_Units=[5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,24,25,27,28,29,30,31,32,...
    %33,34,35,42,44,48,55,56,59,61];
% Results path for nonbase units
%ResultPath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code/NonBase Unit Encoding Results/';
Num_Neurons=length(Base_Unit_Store(1,1:end));
%% Run Encoding Property Analysis Iteratively for each base unit

for Sine_Amp=1:3; % sinusoidal amplitude regimes
    Neuron=0;
    for neuron=Coh_Base_Units
        Identity=Base_Unit_Store{1,neuron};
        WN_Repeat_Matrix=Base_Unit_Store{2,neuron};
        disp(Identity)
        addpath('NonBase Unit Encoding Results')
        addpath('Base Unit Encoding Properties Results')
        % Analyze by Sine Amplitude
        % Load Actual SP, Predicted SR, and Rel Error
        load(['Average Actual_Predicted SR Rel Err Sine ',num2str(Sine_Amp),'V ',Identity,'.mat'])
        Neuron=Neuron+1;
        Rel_Error_Store(Neuron,Sine_Amp)=Rel_Error; % This is actually the RMSE (root mean squared error)
        AVE_Actual_SR_Store(Neuron,Sine_Amp)=AVE_Actual_SR; % Average recorded spike rate
    end
end
% Save Storage Vectors
save([ResultPath,'Base Quantitative Spike Rate Vectors','.mat'],'Rel_Error_Store','AVE_Actual_SR_Store','-mat')
        
        
        
        
        