%% Constructing the non-linear decision function for isolated neurons
% Created by Brandon Pratt
% Date: 08/2016
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
    %33,34,35,42,44,48,55,56,59,61];
% Results path for nonbase units
ResultPath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code/NonBase Unit Encoding Results/';
Num_Base_Neurons=length(Base_Unit_Store(1,1:end));
%% Run Encoding Property Analysis Iteratively for each base unit
for neuron=Coh_Base_Units
    Identity=Base_Unit_Store{1,neuron}; % Unit name (Moth #, Unit #)
    WN_Repeat_Matrix=Base_Unit_Store{2,neuron};
    disp(Identity)
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
    figure(1)
    Total_Time=Num_Repeats*(Num_Samples_Repeat/FS);
    mean_fire_rate=Num_Spikes(Neuron)/Total_Time; % Scaling factor for NLD
    fire_rate{Neuron}= (ste_hist./pe_hist)*mean_fire_rate; % No longer non-dimensional
    
    subplot(1,2,1)
    bar(bin_centers, [ste_hist', pe_hist'], 'grouped')
    xlabel('$\vec{s}$', 'Interpreter', 'Latex','FontSize',16)
    ylabel('PDF','FontSize',16)
    legend('STE', 'PE')
    title([Identity])
    
    
    subplot(1,2,2)
    bar(Bin_Centers_Store{Neuron},fire_rate{Neuron},'FaceColor',[0.6,0.6,0.6],'EdgeColor','none');
    xlabel('$\vec{s}$', 'Interpreter', 'Latex','FontSize',16)
    ylabel('Predicted Firing Rate(Hz)')
    %ylabel('$\frac{ r(t)}{ \bar{r}}$','Interpreter','LaTex','FontSize',24);
    title([Identity])
    hold on
    % Plot smooth function on NLD
    %pcoeffs_NLD=polyfit(Bin_Centers_Store{Neuron},fire_rate{Neuron},6);
    %Fit=polyval(pcoeffs_NLD,Bin_Centers_Store{Neuron});
    fit_x=Bin_Centers_Store{Neuron}(1):0.0001:Bin_Centers_Store{Neuron}(end);
    NLD_Fit=spline(Bin_Centers_Store{Neuron},fire_rate{Neuron},fit_x);%polyval(pcoeffs_NLD,fit_x);
    NLD_Fit(end)=fire_rate{Neuron}(end);
    plot(fit_x,NLD_Fit,'r.','LineWidth',5)
    hold off
    
    % Find Projection Values at Peak Firing rate
    [Peak_FR,Ifr]=max(fire_rate{Neuron});
    Peak_Proj=Bin_Centers_Store{Neuron}(Ifr);
    
    % Find slope at half max peak
    half_max_fr=Peak_FR/2;
    
    % Closet Index Value to half peak firing rate
    [~,idx]=min(abs(half_max_fr-NLD_Fit));
    proj_half_peak=fit_x(idx);
    % Find the slope at the above half peak firing rate using central
    % difference
    slope=(NLD_Fit(idx+1)-NLD_Fit(idx-1))/(fit_x(idx+1)-fit_x(idx-1));
    
    %% Save NLD Results (Bin Centers and Firing Rate
    save([ResultPath,'NLD',Identity,'.mat'],'Bin_Centers_Store','fire_rate','Peak_FR','Peak_Proj','slope','proj_half_peak','-mat')
    %save([ResultPath,'Mean Firing Rate',Identity,'.mat'],'mean_fire_rate','-mat')
    %% Clear Variables
    clearvars -except neuron Coh_Base_Units Base_Unit_Store ResultPath; close all; clc
end