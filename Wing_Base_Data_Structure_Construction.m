%% Data Structure for Analysis
% Created by Brandon Pratt
% Date: January 2017
clear all; close all; clc;

%% Create Data Structure with the Identity of each moth and neuron specified
Base_Unit_Store{1,1}='M2 N1';Base_Unit_Store{1,2}='M2 N2';
Base_Unit_Store{1,3}='M8 N1';Base_Unit_Store{1,4}='M9 N1';
Base_Unit_Store{1,5}='M10 N1';Base_Unit_Store{1,6}='M10 N2';
Base_Unit_Store{1,7}='M11 N1';Base_Unit_Store{1,8}='M11 N2';
Base_Unit_Store{1,9}='M11 N3';Base_Unit_Store{1,10}='M12 N1';
Base_Unit_Store{1,11}='M12 N2';Base_Unit_Store{1,12}='M14 N1';
Base_Unit_Store{1,13}='M16 N1';Base_Unit_Store{1,14}='M16 N2';
Base_Unit_Store{1,15}='M16 N3';Base_Unit_Store{1,16}='M18 N1';
Base_Unit_Store{1,17}='M20 N1';Base_Unit_Store{1,18}='M20 N2';
Base_Unit_Store{1,19}='M23 N2';Base_Unit_Store{1,20}='M23 N3';
Base_Unit_Store{1,21}='M23 N5';Base_Unit_Store{1,22}='M23 N6';
Base_Unit_Store{1,23}='M24 N2';Base_Unit_Store{1,24}='M24 N3';
Base_Unit_Store{1,25}='M25 N1';Base_Unit_Store{1,26}='M26 N3';
Base_Unit_Store{1,27}='M27 N1';Base_Unit_Store{1,28}='M30 N1';
Base_Unit_Store{1,29}='M32 N1';Base_Unit_Store{1,30}='M32 N2';
Base_Unit_Store{1,31}='M33 N1';
%% Load Spike Train Data
% load spikes train files iteratitively and save into data structure

Neuron_Count=0;
for Moth_Num=[2,8,9,10,11,12,14,16,18,20,23,24,25,26,27,30,32,33]
    load(['Spikes_During_WN_Seg_M',num2str(Moth_Num),'.mat'])
    
    % Determine how neurons are contained (how many units there are for
    % this moth)
    Num_Neurons=size(WN_Repeat_Matrix,3);
    
    % Place spike train during white noise for each neuron into data struct
    for Neuron=1:Num_Neurons
        Neuron_Count=Neuron_Count+1;
        Base_Unit_Store{2,Neuron_Count}=WN_Repeat_Matrix(:,:,Neuron);
    end
    clearvars WN_Repeat_Matrix
end

%% Save Data Structure containing the identity and spike trains of Base Units
save('Wing_Base_Identity_Spike_Trains.mat','Base_Unit_Store','-v7.3')


