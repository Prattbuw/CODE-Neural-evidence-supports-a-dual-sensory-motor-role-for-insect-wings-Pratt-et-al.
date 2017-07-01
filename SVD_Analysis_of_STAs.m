%% Constructing the non-linear decision function for isolated neurons
% Created by Brandon Pratt
% Date: 08/2016
clear all; close all; clc

%% Load Data
% NonBase Spike Trains
% Load Spike Trains and Identity of Units that are Base Idenified
load('Wing_NonBase_Identity_Spike_Trains.mat')
Non_Base_Units=[5,6,7,8,9,10,12,13,15,16,17,18,19,20,21,22,24,25,27,28,29,30,31,32,...
    33,34,35,42,44,48,55,56,59,61];
Non_Base_Unit_Store=Base_Unit_Store;
clear Base_Unit_Store
% Base Units
% Load Spike Trains and Identity of Units that are Base Idenified
load('Wing_Base_Identity_Spike_Trains.mat')
Base_Units=[3,4,5,9,10,11,12,14,19,20,23,27,28,31];

%% Run Encoding Property Analysis Iteratively for each base unit
% Add paths to STA data
addpath('NonBase Unit Encoding Results')
addpath('Base Unit Encoding Properties Results')

%Create STA SVD Matrix
STA_Matrix=zeros(1600,48);
Base_STA_Matrix=zeros(1600,length(Base_Units));
Non_Base_STA_Matrix=zeros(1600,length(Non_Base_Units));
%Creat Prior SVD Matrix
Prior_Matrix=zeros(1600,48);
Base_Prior_Matrix=zeros(1600,length(Base_Units));
Non_Base_Prior_Matrix=zeros(1600,length(Non_Base_Units));

% Load STAs for Base Units first
for base_unit=1:length(Base_Units)
    Base_Identity=Base_Unit_Store{1,Base_Units(base_unit)};
    load(['STA and Std',Base_Identity,'.mat'])
    STA_Matrix(:,base_unit)=STA;
    Base_STA_Matrix(:,base_unit)=STA;
    
    % Build Prior Matrix 
    load(['Prior and Prior Std',Base_Identity,'.mat'])
    Prior_Matrix(:,base_unit)=Prior;
    Base_Prior_Matrix(:,base_unit)=Prior;
    clearvars STA Std_Error Prior Prior_Std_Error
end
%%    
% Load STAs fo Non-localized Units
for nonlocalized_units=1:length(Non_Base_Units)
    Non_Base_Identity=Non_Base_Unit_Store{1,Non_Base_Units(nonlocalized_units)};
    load(['STA and Std',Non_Base_Identity,'.mat'])
    STA_Matrix(:,nonlocalized_units+length(Base_Units))=STA;
    Non_Base_STA_Matrix(:,nonlocalized_units)=STA;
    
    % Build Prior Matrix 
    load(['Prior and Prior Std',Non_Base_Identity,'.mat'])
    Prior_Matrix(:,nonlocalized_units+length(Base_Units))=Prior;
    Non_Base_Prior_Matrix(:,nonlocalized_units)=Prior;
    clearvars STA Std_Error Prior Prior_Std_Error
end
    
%% Perform SVD on STA Matrix

%SVD STA Matrix
[U,S,V] = svd(STA_Matrix,'econ');

%SVD Prior Matrix
[U_Prior,S_Prior,V_Prior] = svd(Prior_Matrix,'econ');

% SVD for Base and Non_Base STAs and Priors
[U_Base,S_Base,V_Base] = svd(Base_STA_Matrix,'econ');
[U_NonBase,S_NonBase,V_NonBase] = svd(Non_Base_STA_Matrix,'econ');

% Prior for each type of neuron
[U_Prior_Base,S_Prior_Base,V_Prior_Base] = svd(Base_Prior_Matrix,'econ');
[U_Prior_NonBase,S_Prior_NonBase,V_NonPrior_Base] = svd(Non_Base_Prior_Matrix,'econ');

% Isolate Singular Values and Eigen Modes 
Dominant_Prior_Singular_Value=max(diag(S_Prior));
Dominant_Prior_Singular_Value_Base=max(diag(S_Prior_Base));
Dominant_Prior_Singular_Value_NonBase=max(diag(S_Prior_NonBase));

Singular_Values=diag(S);
Singular_Values_Base=diag(S_Base);
Singular_Values_NonBase=diag(S_NonBase);

% Most dominant modes
Dominant_Modes=U(:,1:6);
Dominant_Modes_Base=U_Base(:,1:6);
Dominant_Modes_NonBase=U_NonBase(:,1:6);
%% Plot dominant set of modes
Time=linspace(-40,0,1600);
% plot order fo singular values
figure(1)
plot(Singular_Values,'k.','MarkerSize',10)
title('Singular Values')
ylabel('Singular Value')
hold on
plot(1:length(Singular_Values),Dominant_Prior_Singular_Value*ones(1,length(Singular_Values)),'r','LineWidth',3)
legend('Singular Values', 'Singular Value Cutoff');
axis tight
hold off


% Plot Most Dominant Modes
figure(2)
for Mode=1:length(Dominant_Modes(1,:))
    subplot(length(Dominant_Modes(1,:)),1,Mode)
    plot(Time.',Dominant_Modes(:,Mode),'k','LineWidth',3)
    title(['Dominant Mode', num2str(Mode)])
    xlabel('Time(ms)')
    ylabel('Displacement(mm)')
    axis tight
end

% Plot Singular Values for Base and nonlocalized units
figure(3)
subplot(1,3,1)
plot(Singular_Values_Base,'k.','MarkerSize',10)
title('Singular Values of Base Localized Units')
ylabel('Singular Value')
hold on
plot(1:length(Singular_Values_Base),Dominant_Prior_Singular_Value_Base*ones(1,length(Singular_Values_Base)),'r','LineWidth',3)
legend('Singular Values', 'Singular Value Cutoff');
axis tight
hold off

subplot(1,3,2)
plot(Singular_Values_NonBase,'k.','MarkerSize',10)
title('Singular Values of Non Localized Units')
ylabel('Singular Value')
hold on 
plot(1:length(Singular_Values_NonBase),Dominant_Prior_Singular_Value_NonBase*ones(1,length(Singular_Values_NonBase)),'r','LineWidth',3)
legend('Singular Values', 'Singular Value Cutoff');
axis tight
hold off

subplot(1,3,3)
plot(Singular_Values,'k.','MarkerSize',10)
title('Singular Values')
ylabel('Singular Value')
hold on
plot(1:length(Singular_Values),Dominant_Prior_Singular_Value*ones(1,length(Singular_Values)),'r','LineWidth',3)
legend('Singular Values', 'Singular Value Cutoff');
axis tight
hold off

% Eigen Modes of both Base and Non-localized Units
figure(4)
img_count=0;
for Mode=1:length(Dominant_Modes(1,:))
    img_count=img_count+1;
    % Plot dominant modes for Base units
    subplot(length(Dominant_Modes(1,:)),3,img_count)
    plot(Time.',Dominant_Modes_Base(:,Mode),'k','LineWidth',3)
    title(['Dominant Mode of Base Units ', num2str(Mode)])
    xlabel('Time(ms)')
    ylabel('Displacement(mm)')
    axis tight
    
    % Plot dominant modes for non-localized units
    img_count=img_count+1;
    subplot(length(Dominant_Modes(1,:)),3,img_count)
    plot(Time.',Dominant_Modes_NonBase(:,Mode),'k','LineWidth',3)
    title(['Dominant Mode of Nonlocalized Units ', num2str(Mode)])
    xlabel('Time(ms)')
    ylabel('Displacement(mm)')
    axis tight
    
    % Plot Combined dominant modes
    img_count=img_count+1;
    subplot(length(Dominant_Modes(1,:)),3,img_count)
    plot(Time.',Dominant_Modes(:,Mode),'k','LineWidth',3)
    title(['Dominant Mode Combined ', num2str(Mode)])
    xlabel('Time(ms)')
    ylabel('Displacement(mm)')
    axis tight
end
    