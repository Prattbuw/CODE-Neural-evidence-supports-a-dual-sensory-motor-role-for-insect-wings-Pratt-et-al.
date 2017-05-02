%% Build Spike Trains for each highly SR coherent neuron
% Created by: Brandon Pratt
% Date: January 2017
clear all; close all; clc

%% Load Data
addpath('Non-Base Units')
% Load Time Stamps of spiking
Time_Stamps=dlmread('M33_Sorted.txt',',',1,0);
Neuron_Time_Stamps=Time_Stamps(:,1:end-1);

%% Now that the Stimulus is reconstructed, Setup Spike Timestamps for Analysis
FS=4e4; % Sampling Rate at 40 kKz
% Convert timestamps into time samples/indices
Neuron_Spike_Time_Samples=Neuron_Time_Stamps.*FS;

% Indice in which WN Stimulus Starts (Check to see if there is a step)
% If file has a step to it add 240000 samples 
%Step_Length=640000;
WN_Start_Indice=21120004;%+Step_Length;
% Account for motor start up delay
Delay=FS.*1;
WN_Start_Indice=WN_Start_Indice+Delay;

% Whitenoise matrix to be filled with spike times
Num_Neurons=length(Neuron_Spike_Time_Samples(1,:));% Number of units isolated from spike sorting for this moth
Num_Repeats=30; %Number of white noise repeats
Duration_of_Repeat=10;% duration of each repeat in seconds
Num_Samples_Repeat=FS*Duration_of_Repeat;% number of samples in each repeat
Num_Samples_WN_Seg=Num_Samples_Repeat*10;% Number of samples of 10 repeats of white noise
WN_Repeat_Matrix=zeros(Num_Samples_Repeat,Num_Repeats,1); %Adjust for the number of neurons
Delay_Btwn_WN_Seg=3*FS;% Time delay between each white noise repeat in seconds


%% Spike Train Construction Method 1 (Compare to below method) Much Faster
Neuron_Count=0;
for Neuron_Num=2; %:Num_Neurons
    Start_Idx=WN_Start_Indice;
    Neuron_Count=Neuron_Count+1;
    for Repeat_Num=1:Num_Repeats
        disp([' Repeat',num2str(Repeat_Num)])
        Spike_Window=Start_Idx:Start_Idx+Num_Samples_Repeat-1;
        Idx_Spikes_Repeat=find(round(Neuron_Spike_Time_Samples(:,Neuron_Num))>=Spike_Window(1)...
            & round(Neuron_Spike_Time_Samples(:,Neuron_Num))<=Spike_Window(end));
        
        % Find relative spike index during WN repeat
        Rel_Spike_Idx_Repeat=round(Neuron_Spike_Time_Samples(Idx_Spikes_Repeat,Neuron_Num))-Spike_Window(1);
        %Rel_Spike_Idx_Repeat(1)=Rel_Spike_Idx_Repeat(1)+1;
       
        %This fills in spike idx location
        WN_Repeat_Matrix(Rel_Spike_Idx_Repeat,Repeat_Num,Neuron_Count)=1;
            
            if Start_Idx==WN_Start_Indice+Num_Samples_WN_Seg-Num_Samples_Repeat
                Start_Idx=Start_Idx+Delay_Btwn_WN_Seg;  
            end
            
            if Start_Idx==WN_Start_Indice+((2*Num_Samples_WN_Seg)+Delay_Btwn_WN_Seg)-Num_Samples_Repeat
                Start_Idx=Start_Idx+Delay_Btwn_WN_Seg;   
            end
            % Adjust Window Length
        Start_Idx=Start_Idx+Num_Samples_Repeat;
        
        clearvars Idx_Spikes_Repeat Rel_Spike_Idx_Repeat
    end
end

%% Save WN Repeat Matrix with Corresponding Spike Indices
Filepath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code';
save([Filepath,'/NonBase Spike Trains/NonBase Spike Train_M33.mat'],'WN_Repeat_Matrix','-mat') %Alter WN_Repeat_Matrix possible

% %% Method 2: Construct Spike Sample locations in each sample (Slower
% method)
% for Neuron_Num=[1,2] %:Num_Neurons
%     Indice_Count=0;
%     for Repeat_Num=1:Num_Repeats
%         disp([' Repeat',num2str(Repeat_Num)])
%         for Repeat_Sample_Num=1:Num_Samples_Repeat
%                % ' Index',num2str(Repeat_Sample_Num)])
%             %disp(['Neuron=',num2str(Neuron_Num), ' Repeat',num2str(Repeat_Num),...
%                % ' Index',num2str(Repeat_Sample_Num)])
%             Indice_Count=Indice_Count+1;
%             WN_Sample_Num=WN_Start_Indice+Indice_Count;
%             
%             if any(round(Neuron_Spike_Time_Samples(:,Neuron_Num))==WN_Sample_Num)
%                 WN_Repeat_Matrix(Repeat_Sample_Num,Repeat_Num,Neuron_Num)=1;
%             else
%                 WN_Repeat_Matrix(Repeat_Sample_Num,Repeat_Num,Neuron_Num)=0;
%             end
%             
%             if Indice_Count==Num_Samples_WN_Seg
%                 Indice_Count=Indice_Count+Delay_Btwn_WN_Seg;
%             end
%             
%             if Indice_Count==(2*Num_Samples_WN_Seg)+Delay_Btwn_WN_Seg
%                 Indice_Count=Indice_Count+Delay_Btwn_WN_Seg;
%             end
%         end
%     end
% end