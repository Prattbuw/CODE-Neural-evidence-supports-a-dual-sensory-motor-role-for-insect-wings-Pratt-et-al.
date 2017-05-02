%% Data Structure for Analysis
% Created by Brandon Pratt
% Date Janurary 2017
clear all; close all; clc;

%% Create Data Structure with the Identity of each moth and neuron specified
% In this case, Base_Unit_Store does not refer to storing only the units
% localized to the base of the wing, but rather units that could not be
% localized. Not changing the variable unit made for less alterations in
% the analyses scripts
Base_Unit_Store{1,1}='M1 N1';Base_Unit_Store{1,2}='M1 N2';
Base_Unit_Store{1,3}='M1 N3';Base_Unit_Store{1,4}='M1 N4';
Base_Unit_Store{1,5}='M3 N1';Base_Unit_Store{1,6}='M4 N1';
Base_Unit_Store{1,7}='M4 N2';Base_Unit_Store{1,8}='M5 N1';
Base_Unit_Store{1,9}='M5 N2';Base_Unit_Store{1,10}='M6 N1';
Base_Unit_Store{1,11}='M6 N2';Base_Unit_Store{1,12}='M6 N3';
Base_Unit_Store{1,13}='M7 N1';Base_Unit_Store{1,14}='M7 N2';
Base_Unit_Store{1,15}='M7 N3';Base_Unit_Store{1,16}='M8 N2';
Base_Unit_Store{1,17}='M8 N3';Base_Unit_Store{1,18}='M8 N4';
Base_Unit_Store{1,19}='M9 N2';Base_Unit_Store{1,20}='M10 N3';
Base_Unit_Store{1,21}='M12 N3';Base_Unit_Store{1,22}='M12 N4';
Base_Unit_Store{1,23}='M13 N1';Base_Unit_Store{1,24}='M13 N2';
Base_Unit_Store{1,25}='M14 N2';Base_Unit_Store{1,26}='M14 N3';
Base_Unit_Store{1,27}='M14 N4';Base_Unit_Store{1,28}='M15 N1';
Base_Unit_Store{1,29}='M17 N1';Base_Unit_Store{1,30}='M17 N2';
Base_Unit_Store{1,31}='M17 N3';Base_Unit_Store{1,32}='M18 N2';
Base_Unit_Store{1,33}='M18 N3';Base_Unit_Store{1,34}='M19 N1';
Base_Unit_Store{1,35}='M19 N2';Base_Unit_Store{1,36}='M19 N3';
Base_Unit_Store{1,37}='M20 N3';Base_Unit_Store{1,38}='M21 N1';
Base_Unit_Store{1,39}='M21 N2';Base_Unit_Store{1,40}='M21 N3';
Base_Unit_Store{1,41}='M22 N1';Base_Unit_Store{1,42}='M22 N2';
Base_Unit_Store{1,43}='M23 N1';Base_Unit_Store{1,44}='M23 N4';
Base_Unit_Store{1,45}='M24 N1';Base_Unit_Store{1,46}='M25 N2';
Base_Unit_Store{1,47}='M25 N3';Base_Unit_Store{1,48}='M26 N1';
Base_Unit_Store{1,49}='M26 N2';Base_Unit_Store{1,50}='M27 N2';
Base_Unit_Store{1,51}='M27 N3';Base_Unit_Store{1,52}='M28 N1';
Base_Unit_Store{1,53}='M28 N2';Base_Unit_Store{1,54}='M28 N3';
Base_Unit_Store{1,55}='M29 N1';Base_Unit_Store{1,56}='M29 N2';
Base_Unit_Store{1,57}='M30 N2';Base_Unit_Store{1,58}='M30 N3';
Base_Unit_Store{1,59}='M31 N1';Base_Unit_Store{1,60}='M31 N2';
Base_Unit_Store{1,61}='M31 N3';Base_Unit_Store{1,62}='M31 N4';
Base_Unit_Store{1,63}='M32 N3';Base_Unit_Store{1,64}='M33 N1';


%% Load Spike Train Data
% load spikes train files iteratitively and save into data structure

Neuron_Count=0;
for Moth_Num=[1,3,4,5,6,7,8,9,10,12,13,14,15,17,18,19,20,21,22,23,24,25,...
        26,27,28,29,30,31,32,33]
    load(['NonBase Spike Train_M',num2str(Moth_Num),'.mat'])
    
    % Determine how neurons are contained (how many units there are for a
    % moth)
    Num_Neurons=size(WN_Repeat_Matrix,3);
    
    % Place spike train during white noise for each neuron into data struct
    for Neuron=1:Num_Neurons
        Neuron_Count=Neuron_Count+1;
        Base_Unit_Store{2,Neuron_Count}=WN_Repeat_Matrix(:,:,Neuron);
    end
    clearvars WN_Repeat_Matrix
end

%% Save Data Structure containing the identity and spike trains of Base Units
save('Wing_NonBase_Identity_Spike_Trains.mat','Base_Unit_Store','-v7.3')


