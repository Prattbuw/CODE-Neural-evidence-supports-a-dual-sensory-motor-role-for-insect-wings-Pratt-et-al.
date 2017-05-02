%% Determine the Average smoothed and predicted spike rate for each Highly Coherent Neuron
% Created by Brandon Pratt
%Date: January 2017
clear all; close all; clc

%% Load Data
% Load Spike Trains and Identity of Units that are Base Idenified
%load('Wing_Base_Identity_Spike_Trains.mat')
% Results path for base units
%ResultPath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code/Base Unit Encoding Properties Results/';
%Coh_Base_Units=[3,4,5,9,10,11,12,14,19,20,23,27,28,31];

% NonBase Spike Trains
% Load Spike Trains and Identity of Units that are Base Idenified
load('Wing_NonBase_Identity_Spike_Trains.mat')
Coh_Base_Units=[5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,24,25,27,28,29,30,31,32,...
33,34,35,42,44,48,55,56,59,61];
% Results path for nonbase units
ResultPath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code/NonBase Unit Encoding Results/';
Num_Base_Neurons=length(Base_Unit_Store(1,1:end));
%% Run Encoding Property Analysis Iteratively for each base unit
for neuron=Coh_Base_Units
    Identity=Base_Unit_Store{1,neuron}; % Unit Name (Moth#, Neuron #)
    WN_Repeat_Matrix=Base_Unit_Store{2,neuron};
    disp(Identity)
    Moth_Num=str2double(Identity(2:3));
    % Load time stamps of spikes for each neuron from spike sorting
    addpath('Base Units')
    addpath('Non-Base Units')
    % Load Time Stamps
    Time_Stamps=dlmread(['M',num2str(Moth_Num),'_Sorted.txt'],',',1,0); % Load spike timestamps
    Neuron_Time_Stamps=Time_Stamps(:,str2double(Identity(end)));
    Neuron=1;
    % Load time stamps of spikes for each neuron from spike sorting
    addpath('NonBase Unit Encoding Results')
    addpath('Base Unit Encoding Properties Results')
    
    % Load STA, STE, and PSE for each neuron use moth Idenitity to do so
    % Load STA
    load(['STA and Std',Identity,'.mat'])
    
    % Load STE
    load(['STE',Identity,'.mat'])
    
    % Load PSE
    load(['PSE',Identity,'.mat'])
    
    %% 1D Nonlinear Decision Function
    % Set Parameters
    FS=4e4;
    % Convert timestamps into time samples/indices
    Neuron_Spike_Time_Samples=Neuron_Time_Stamps.*FS;
    Num_Spikes=length(STE(:,1));
    Num_Repeats=30;
    Duration_of_Repeat=10;
    Num_Samples_Repeat=FS*Duration_of_Repeat;
    Num_Samples_WN_Seg=Num_Samples_Repeat*10;
    % Calculate the non-linear decison function for each neuron
    for Spikes=1:Num_Spikes
        STA_Norm=STA(Neuron,:)/norm(STA(Neuron,:));%Jared Norm
        % Calculate P(s|spike)
        STE_Norm=STE(Spikes,:)/norm(STE(Spikes,:));
        STE_Projection{Neuron}(1,Spikes)=dot(STA_Norm,STE_Norm);
        % Calculate P(s)
        PSE_Norm=Prior_Ensemble{Neuron}(Spikes,:)/norm(Prior_Ensemble{Neuron}(Spikes,:));
        Prior_Projection{Neuron}(1,Spikes)=dot(STA_Norm,PSE_Norm);
        %Orginal Normalization
        %STE_Projection{Neuron}(1,Spikes)=dot(STA(Neuron,:),STE_Store{Neuron}(Spikes,:))./dot(STA(Neuron,:),STA(Neuron,:));
    end
    [mu, sigma] = normfit(Prior_Projection{Neuron});
    bin_width = 0.4*sigma;
    num_bins = round((max(Prior_Projection{Neuron}) - min(Prior_Projection{Neuron})) / bin_width);
    
    [pe_hist, bin_centers] = hist(Prior_Projection{Neuron}, num_bins);
    [ste_hist, bin_centers] = hist(STE_Projection{Neuron}, bin_centers) ;
    Bin_Centers_Store{Neuron}=bin_centers;
    
    % Normalize both histograms
    ste_hist = ste_hist/sum(ste_hist);
    pe_hist = pe_hist/sum(pe_hist);
    
    % Quotient gives r(t)/r_avg
    Total_Time=Num_Repeats*(Num_Samples_Repeat/FS);
    mean_fire_rate=Num_Spikes(Neuron)/Total_Time; % Scaling factor for NLD
    fire_rate{Neuron}= (ste_hist./pe_hist)*mean_fire_rate; % No longer non-dimensional
    
    
    %% Choose Stimulus Type and Construct Spike Train
    % Choose Stimulus type (i.e. sinusodial amplitude)
    for Sine_Struct_Num=1:3
        Sine_Struct=load(['sine',num2str(Sine_Struct_Num),'V_corrected.mat']); %Stim for Sine1V
        %Sine_Struct=load('sine2V_corrected.mat'); %Stim for Sine2V
        %Sine_Struct=load('sine3V_corrected.mat'); %Stim for Sine3V
        % Convert Sine from Volts into mm
        load('Quad_V_MM_Conversion.mat')
        MM_Quadratic=@(V) p4(1).*V.^2 +p4(2).*V + p4(3);
        Stim=Sine_Struct.stimulus; %Stimulus
        % Conversion from Volts to mm
        Stim=MM_Quadratic(Stim);
        t=Sine_Struct.t;
        Stim_Template=Stim(1:1600*2); %For prediction 3 sine cycles start at index 1600
        %Stim=UpSample_WN_Stim(1:end-400);
        %t=linspace(0,length(WN_Stim)/1000,length(UpSample_WN_Stim));
        
        % FIND Min Index
        %[V,idx]=min(Stim(1:1600));
        %% Predict spikes given a neurons STA and NLD
        
        for j=1:length(Stim_Template)
            disp(num2str(j))
            STA_Norm=STA(Neuron,:)/norm(STA(Neuron,:));
            Stim_Seg=Stim_Template(j:j+1599);% slide through stim segements of 1600 samples
            Stim_Norm=Stim_Seg/norm(Stim_Seg); %Norm stim
            Stim_Projection=dot(STA_Norm,Stim_Norm); %Projection onto stim segment
            % Determine what bin is closest to the stim projection
            %delta_proj=Bin_Centers_Store{Neuron}-repmat(Stim_Projection,1,length(Bin_Centers_Store{Neuron}));
            %[Min_Value, Min_idx]=min(abs(delta_proj));
            % Predicted Fire Rate in time
            pcoeffs=polyfit(Bin_Centers_Store{Neuron},fire_rate{Neuron},1);
            Predicted_Fire_Rate_Proj{Neuron}(j)=polyval(pcoeffs,Stim_Projection);
            %Predicted_Fire_Rate_Proj{Neuron}(j)=interp1(Bin_Centers_Store{Neuron},fire_rate{Neuron},Stim_Projection);
            % Deal with last 1600 boundary effect
            if j==length(Stim_Template)-1600
                break;
            end
            %clearvars delta_proj
        end
        
        %% Construct Spike Train for Sine Stimulus in the form of a raster
        % Choose corresponding Start indice
        Sine_Start_Indice(1)=20600002; %1V
        Sine_Start_Indice(2)=20760002; %2V
        Sine_Start_Indice(3)=20920003; %3V
        
        % Choose Corresponding End indice
        %Sine_End_Indice=20760001; %1V
        %Sine_End_Indice=20920001; %2V
        %Sine_End_Indice=21080002; %3V
        
        % Determine how many sine repeats there are
        Sine_Window=1600;
        Num_Sine_Repeats=length(Stim)/Sine_Window; %1600 is 40ms sine presentation
        Sine_Phase_Stim=Stim(1:1600);
        
        %Construct Spike Train during Sine
        
        indice_count=0;
        for Sine_Repeat=1:Num_Sine_Repeats
            disp(['Sine Repeat ',num2str(Sine_Repeat)])
            for Sample_Num=1:Sine_Window;
                indice_count=indice_count+1;
                Sine_Sample_Num=Sine_Start_Indice(Sine_Struct_Num)+indice_count;
                if any(round(Neuron_Spike_Time_Samples(:,Neuron))==Sine_Sample_Num)
                    Sine_Spike_Train{Neuron}(Sample_Num,Sine_Repeat)=1;
                else
                    Sine_Spike_Train{Neuron}(Sample_Num,Sine_Repeat)=0;
                end
            end
        end
        
        %% Bin Sine Spike Trains
        Sine_bin_index_width=50; %Adjust Bin Size (factors of 1600 only)
        %2, 5, 10, 20, 40, 50 (Good),64, 80(BEST),100(Better),160 (Best)
        nbins=Sine_Window/Sine_bin_index_width;
        Bin_Time=.04/nbins*Num_Sine_Repeats;  % .040s sine cycle length (25 Hz) % Time Each bin represents
        % Multiple by 100 because of the number of sine repeats
        % Choose Neuron
        
        bin_count=0;
        for Bin_Num=1:nbins
            bin_count=bin_count+1;
            % for bin widths greater than 1
            Sine_Hist{Bin_Num}=Sine_Spike_Train{Neuron}(bin_count:bin_count+Sine_bin_index_width-1,:);
            bin_count=bin_count+Sine_bin_index_width-1;
        end
        
        % Number of Spikes, mean, and std for each bin
        
        for Bin_Num=1:nbins
            Num_Spikes_Bin(Bin_Num)=sum(sum(Sine_Hist{Neuron,Bin_Num}));
            Mean_Spike_Bin(Bin_Num)=mean2(Sine_Hist{Neuron,Bin_Num});
            Std_Spike_Bin(Bin_Num)=std2(Sine_Hist{Neuron,Bin_Num});
        end
        
        % Plot Bar Graph of Actual Spike Rate
        Bin_Centers=0:Sine_bin_index_width:Sine_Window-Sine_bin_index_width;
        Err_Center=Sine_bin_index_width/2:Sine_bin_index_width:Sine_Window;
        Actual_Spike_Rate=Num_Spikes_Bin./Bin_Time;
        
        % Construct Guassian window
        guass_length=length(Actual_Spike_Rate);
        x_guass=1:guass_length;
        x0_guass=guass_length/2;
        A=1;
        W=6; % standard deviation is W/sqrt(2)
        y_guass=A*exp(-((x_guass-x0_guass)/W).^2);
        A=1/trapz(y_guass);
        y_cal=y_guass*A;
        
        % Covolve Actual Spike rate with guassian kernal
        Smooth_Actual_Spike_Rate=conv([Actual_Spike_Rate...
            ,Actual_Spike_Rate,Actual_Spike_Rate],y_cal,'same');
        
        % window for conv x units for Convovled Spike Rate
        Offset=length(Actual_Spike_Rate)+1;
        Conv_Window=2*length(Actual_Spike_Rate);%+length(y_cal)-1;
        Conv_SR=Offset:Conv_Window;
        x_sampled=linspace(1,1600,length(Conv_SR));
        
        % Interpolate smoothed spike rate
        x_interp=linspace(1,1600,1600);
        y_smooth=spline(x_sampled,Smooth_Actual_Spike_Rate(Conv_SR),x_interp);
        
        % Average Smoothed Actual Spike Rate
        AVE_Actual_SR=mean(Smooth_Actual_Spike_Rate(Conv_SR));
        
        % Average Predicted Spike Rate
        AVE_Pred_SR=mean(Predicted_Fire_Rate_Proj{Neuron});
        
        % Relative Error (RMSE)
        Rel_Error=sqrt(mean((y_smooth-Predicted_Fire_Rate_Proj{Neuron}).^2));
        
        %% Save Average predicted and actual SR and relative error Results (Bin Centers and Firing Rate
        save([ResultPath,'Average Actual_Predicted SR Rel Err Sine ',num2str(Sine_Struct_Num),'V ',Identity,'.mat'],'AVE_Actual_SR','AVE_Pred_SR','Rel_Error','-mat')
        
        %% Clear Variables
        clearvars Sine_Hist Num_Spikes_Bin Mean_Spike_Bin Std_Spike_Bin bin_count...
            Predicted_Fire_Rate_Proj Stim_Projection Stim_Norm STA_Norm Sine_Struct Stim Stim_Template...
            t Smooth_Actual_Spike_Rate Actual_Spike_Rate y_guass x_guass y_cal Conv_SR x_sampled
    end
    
    %% Clear Variables
    clearvars -except neuron Coh_Base_Units Base_Unit_Store ResultPath; close all; clc
end