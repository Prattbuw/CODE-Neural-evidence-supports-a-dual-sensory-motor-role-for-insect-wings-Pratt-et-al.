%% Determine the Encoding Properties for each Identified Highly Coherent Neuron
% Created by: Brandon Pratt
% Date: Novemeber 2016
clear all; close all; clc

%% Load Data
% Load upsampled generalized base stimulus from the tip to base stimulus
% transform
load('UpSampled_Generalized Base Displacement.mat')
UpSample_WN_Stim=Upsample_y_Base;
clear Upsample_y_Base

% Load generalized base stimulus that has not been upsampled
load('Generalized Base Displacement.mat')
WN_Stim=Upsample_y_Base;

% Load Spike Trains and Identity of Units that are Base Idenified
load('Wing_Base_Identity_Spike_Trains.mat')
Coh_Base_Units=28; %[3,4,5,9,10,11,12,14,19,20,23,27,28,31];
Num_Base_Neurons=length(Base_Unit_Store(1,1:end));
%% Run Encoding Property Analysis Iteratively for each base unit
for neuron=Coh_Base_Units
    Identity=Base_Unit_Store{1,neuron};
    WN_Repeat_Matrix=Base_Unit_Store{2,neuron};
    disp(Identity)
    Moth_Num=str2double(Identity(2:3));
    % Load time stamps of spikes for each neuron from spike sorting
    addpath('Base Units')
    % Load Time Stamps
    Time_Stamps=dlmread(['M',num2str(Moth_Num),'_Sorted.txt'],',',1,0);
    Neuron_Time_Stamps=Time_Stamps(:,str2double(Identity(end)));
    % Load Spike Train of Units that are Base Idenified and have significant SR
    % Coherence
    % Name variable WN_Repeat_Matrix
    
    % %% UpSample Stimulus Segment by 40x
    % % Upsample to match the stimulus sampling with the spike timestamp sampling
    % Time_Reconstructed_Stimulus=linspace(0,length(WN_Stim)/1000,...
    %     length(WN_Stim));
    % % Upsample Stimulus
    % Upsample_Time_Reconstructed_Stim=linspace(0,length(WN_Stim)/1000,...
    %     length(WN_Stim)*40);
    % UpSample_WN_Stim=spline(Time_Reconstructed_Stimulus,WN_Stim,Upsample_Time_Reconstructed_Stim).';
    
    %% Parameters
    FS=4e4; % Sampling Rate at 40 kKz
    % Convert timestamps into time samples/indices
    Neuron_Spike_Time_Samples=Neuron_Time_Stamps.*FS;
    % Whitenoise matrix to be filled with spike times
    %Num_Neurons=length(Neuron_Spike_Time_Samples(1,:));
    Num_Repeats=30;
    Duration_of_Repeat=10;
    Num_Samples_Repeat=FS*Duration_of_Repeat;
    Num_Samples_WN_Seg=Num_Samples_Repeat*10;
    %WN_Repeat_Matrix=zeros(Num_Samples_Repeat,Num_Repeats,Num_Neurons);
    Delay_Btwn_WN_Seg=3*FS;
    Neuron=1;
    
    %% Plot Raster for each neuron over white noise repeats
    Raster_Time=linspace(0,length(WN_Stim)/1000,...
        length(UpSample_WN_Stim));
    Raster_Stim=18*UpSample_WN_Stim+40.5; % Base
    
    % Raster Plot
    
    for Repeat=1:Num_Repeats
        figure(1)
        Ind_Spikes=find(WN_Repeat_Matrix(1:end,Repeat)==1);
        Raster_Row_Value=((Num_Repeats*ones(1,length(Ind_Spikes)))+2)-Repeat;
        %subplot(1,Num_Neurons,Neuron)
        plot(Raster_Stim,'k-','LineWidth',0.01) %Plot WN stimulus over raster
        hold on;
        %plot(Raster_Stim_Tip,'k-','LineWidth',0.01); % PLot Tip Stimulus
        plot(Ind_Spikes,Raster_Row_Value,'k.','MarkerSize',10);
        %hold on; %Uncomment to run raster without stimulus
        ylim([0, 45])
        xlim([0 length(Raster_Time)])
        xlabel('Sample Number')
        ylabel('White Noise Repeat#')
        title([Identity])
        clearvars Ind_Spikes Raster_Row_Value
    end
    
    FigPath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code/Base Unit Encoding Properties Figs';
    savefig(figure(1),[FigPath,'/White Noise Rasters/Raster',Identity,'.fig'])
    %% Constructing STE and STA for each Neuron
    Sample_Rate=4e4; %Sample per Second
    STS_Time_Window=0.040; %Seconds
    Sample_Window=(Sample_Rate.*STS_Time_Window)-1;% 1599 samples
    Total_Sample_Window=Sample_Rate*STS_Time_Window; % 1600 samples
    %Calulcate the STE for each
    
    Spike_Count=0;
    for Repeat=1:Num_Repeats
        Ind_Spikes=find(WN_Repeat_Matrix(1:end,Repeat)==1); % Acess spike train matrix and find spikes
        Ind_Spikes=Ind_Spikes(find(Ind_Spikes>1600 & Ind_Spikes<length(UpSample_WN_Stim)));
        for j=1:length(Ind_Spikes)
            Spike_Count=Spike_Count+1;
            STS=UpSample_WN_Stim((Ind_Spikes(j)-Sample_Window):Ind_Spikes(j)); % Determine stimulus preceeding the spike
            STE(Spike_Count,1:Total_Sample_Window)=STS.'; % Build STE matrix
            clearvars STS
        end
        clearvars Ind_Spikes
    end
    % Construct the spike trigger average
    STA(Neuron,1:Total_Sample_Window)=mean(STE,1);
    % Standard deviation
    Std_Error(Neuron,1:Total_Sample_Window)=std(STE, 0, 1 );
    % Number of spikes
    Num_Spikes(Neuron)=Spike_Count;
    % STE matrix
    STE_Store{Neuron}=STE;
    
    ResultPath='/Users/brandonpratt/Dropbox/Brandon/Moth Wing Paper/WingPaper_WorkingVersion/Code for Paper/WN Stimulus Analysis Code';
    save([ResultPath,'/Base Unit Encoding Properties Results/STA and Std',Identity,'.mat'],'STA','Std_Error','-mat')
    save([ResultPath,'/Base Unit Encoding Properties Results/STE',Identity,'.mat'],'STE','-mat')
    %% Plot STA with bounded error for each neuron
    Time_Window_Vec=linspace(-40,0,1600);
    Num_Samples=1:40;
    
    figure(2)
    % Plot STA of Pricipal Strain
    %subplot(1,Num_Neurons,Neuron)
    hold on
    
    h =    fill(    [Time_Window_Vec, fliplr(Time_Window_Vec)], [STA(Neuron,:)-Std_Error(Neuron,:)...
        fliplr(STA(Neuron,:)+Std_Error(Neuron,:))],...
        [0.8,0,0], 'EdgeColor','none');   % %             [0.9,0.9,1], 'EdgeColor','none');
    set(h,'facealpha',.2);
    plot(Time_Window_Vec, squeeze(STA(Neuron,:)),'r')
    xlabel('Time(ms)')
    ylabel('Base Displacement(mm)')
    title([Identity])
    hold off
    
    savefig(figure(2),[FigPath,'/STAs/STA',Identity,'.fig'])
    
    
    %% Permutations of spikes to construct prior ensemble
    Num_Perms=40;
    % Randomly permute the spikes
    
    for Repeat=1:Num_Repeats
        disp([' Repeat',num2str(Repeat)])
        Perm_Spike_Count=0;
        Perm_Indices=[randperm(length(UpSample_WN_Stim)-1600)].';
        Perm_Indices_Correct=[(1:1600).';(Perm_Indices+1600*ones(length(Perm_Indices),1))];
        Ind_Spikes=find(WN_Repeat_Matrix(Perm_Indices_Correct,Repeat,Neuron)==1);
        Ind_Spikes=Ind_Spikes(find(Ind_Spikes>1600 & Ind_Spikes<length(UpSample_WN_Stim)));
        for j=1:length(Ind_Spikes)
            Perm_Spike_Count=Perm_Spike_Count+1;
            STS=UpSample_WN_Stim((Ind_Spikes(j)-Sample_Window):Ind_Spikes(j));
            Perm_STE(Perm_Spike_Count,1:Total_Sample_Window)=STS.';
            clearvars STS
        end
        PERM_STE{Neuron,Repeat}=Perm_STE;
        clearvars Perm_STE
    end
    
    
    %% Create Prior Ensembles for each neuron
    %load('PERM_STE_M16.mat');
    % Concatenate the repeat spike cells for each neuron
    
    Prior_Ensemble{Neuron}=PERM_STE{Neuron,1};
    for Repeat=2:Num_Repeats
        Prior_Ensemble{Neuron}=[Prior_Ensemble{Neuron};PERM_STE{Neuron,Repeat}];
    end
    
    % Save Permutated STE for Prior Construction
    save([ResultPath,'/Base Unit Encoding Properties Results/PSE',Identity,'.mat'],'Prior_Ensemble','-mat')
    % Calculate the Prior and std error of prior
    
    Prior(Neuron,1:Total_Sample_Window)=mean(Prior_Ensemble{Neuron},1);
    Prior_Std_Error(Neuron,1:Total_Sample_Window)=std(Prior_Ensemble{Neuron}, 0, 1 );
    
    % Save Permutated STE for Prior Construction
    save([ResultPath,'/Base Unit Encoding Properties Results/Prior and Prior Std',Identity,'.mat'],'Prior','Prior_Std_Error','-mat')
    
    %% Plot Prior with bounded error for each neuron
    Time_Window_Vec=linspace(-40,0,1600);
    Num_Samples=1:40;
    % Plot STA of Pricipal Strain
    % subplot(1,Num_Neurons,Neuron)
    figure(3)
    hold on
    
    h =    fill(    [Time_Window_Vec, fliplr(Time_Window_Vec)], [Prior(Neuron,:)-Prior_Std_Error(Neuron,:)...
        fliplr(Prior(Neuron,:)+Prior_Std_Error(Neuron,:))],...
        [0.8,0,0], 'EdgeColor','none');   % % [0.9,0.9,1], 'EdgeColor','none');
    set(h,'facealpha',.2);
    plot(Time_Window_Vec, squeeze(Prior(Neuron,:)),'r')
    xlabel('Time(ms)')
    ylabel('Base Displacement(mm)')
    title([Identity])
    hold off
    
    savefig(figure(3),[FigPath,'/Prior/Prior',Identity,'.fig'])
    
    %% 2D Feature Detection E1 and E2 
    
    %Covariance Matrix of the STE
    cov_STE=cov(STE_Store{Neuron});
    % Covariance matrix of the PSE
    cov_PRIOR=cov(Prior_Ensemble{Neuron});
    % Take the difference of the the two COV matrices
    COV_DIFF=cov_STE-cov_PRIOR;
    % Eigenvale decomposion
    [eigen_vector,eigen_value]=eig(COV_DIFF);
    % Find the domiant eigenvalues and eigenvectors
    [abs_max,I] = max(abs(eigen_value),[],1);
    [Sorted_eigenvalues{Neuron},Idenity]=sort(abs_max,'descend');
    E1{Neuron}=Sorted_eigenvalues{1}(1);
    E2{Neuron}=Sorted_eigenvalues{1}(2);
    Eigen_Vector_Sorted=eigen_vector(:,Idenity); %Rearrange eigen vectors
    E1_eigenvector{Neuron}=Eigen_Vector_Sorted(:,1);
    E2_eigenvector{Neuron}=Eigen_Vector_Sorted(:,2);
    %clearvars cov_STE cov_PRIOR COV_DIFF eigen_vector eigen_value Eigen_Vector_Sorted
    save([ResultPath,'/Base Unit Encoding Properties Results/E1 and E2',Identity,'.mat'],'E1','E2','E1_eigenvector','E2_eigenvector','-mat')
    
    %% Plot Sorted Eigen Values and E1 and E2 for each neuron
    % Plot Eigenvalues for each neuron
    Time_Window_Vec=linspace(-40,0,1600);
    figure(4)
    %subplot(Num_Neurons,1,Neuron)
    plot(Sorted_eigenvalues{Neuron}(1:50),'k.') %plot first 50 eigenvalues
    xlabel('Sample(#)')
    ylabel('Eigen value')
    title([Identity])
    savefig(figure(4),[FigPath,'/Eigenvalues/Eigenvalues',Identity,'.fig'])
    
    % Plot E1 and E2 for each Neuron
    figure(5)
    subplot(1,2,1)
    plot(Time_Window_Vec,E1_eigenvector{Neuron},'b','LineWidth',2) %plot E1
    hold on
    plot(Time_Window_Vec,STA(Neuron,:),'k-','LineWidth',2)
    xlabel('Time(ms)')
    ylabel('Eigen vector')
    title(['E1 ' Identity])
    legend('E1','STA','Location','NorthWest')
    hold off
    
    
    subplot(1,2,2)
    plot(Time_Window_Vec,E2_eigenvector{Neuron},'r','LineWidth',2) %plot E2
    hold on
    plot(Time_Window_Vec,STA(Neuron,:),'k-','LineWidth',2)
    xlabel('Time(ms)')
    ylabel('Eigen vector')
    title(['E2 ' Identity])
    legend('E2','STA','Location','NorthWest')
    hold off
    
    savefig(figure(5),[FigPath,'/E1 and E2/E1 and E2',Identity,'.fig'])
    %% 2D projection of E1 and E2 onto the STE and PSE
    
    for Spikes=1:Num_Spikes(Neuron)
        %Calculate the projection for the STE
        E1_Projection_STE{Neuron}(Spikes)=dot(STE_Store{Neuron}(Spikes,:),E1_eigenvector{Neuron});
        E2_Projection_STE{Neuron}(Spikes)=dot(STE_Store{Neuron}(Spikes,:),E2_eigenvector{Neuron});
        
        % Calculate the projection for the prior ensemble
        E1_Projection_PSE{Neuron}(Spikes)=dot(Prior_Ensemble{Neuron}(Spikes,:),E1_eigenvector{Neuron});
        E2_Projection_PSE{Neuron}(Spikes)=dot(Prior_Ensemble{Neuron}(Spikes,:),E2_eigenvector{Neuron});
    end
    
    
    %% Plot the 2D Projections
    
    figure(6)
    % Project the dot product of the prior ensemble and E1 and E2
    %subplot(Num_Neurons,1,Neuron)
    plot(E1_Projection_PSE{Neuron},E2_Projection_PSE{Neuron},'k.')
    hold on
    plot(E1_Projection_STE{Neuron},E2_Projection_STE{Neuron},'r.')
    xlabel('Eig 1','FontSize',16)
    ylabel('Eig 2','FontSize',16)
    title([Identity],'FontSize',16)
    legend('STE', 'Prior Ensemble','Location','NorthEastOutside')
    axis tight
    % axis([-1.5,1.5,-1.5,1.5])
    hold off
    
    savefig(figure(6),[FigPath,'/2D Projections/2D Projections',Identity,'.fig'])
    
    
    %% 2D Feature Detection: Determine Eigenvalues larger than prior Eigenvalues
    % Determine whether E1 and E2 are significant as compared to the PSE
    % derived
    % eigenvalues
    % Details of data structures are above
    cov_STE=cov(STE_Store{Neuron});
    cov_PRIOR=cov(Prior_Ensemble{Neuron});
    [eigen_vector_STE,eigen_value_STE]=eig(cov_STE);
    [eigen_vector_Prior,eigen_value_Prior]=eig(cov_PRIOR);
    [abs_max_STE,I_STE] = max(abs(eigen_value_STE));
    [abs_max_Prior,I_Prior] = max(abs(eigen_value_Prior));
    [Sorted_eigenvalues_STE{Neuron},Idenity_STE]=sort(abs_max_STE,'descend');
    [Sorted_eigenvalues_Prior{Neuron},Idenity_Prior]=sort(abs_max_Prior,'descend');
    E1_STE{Neuron}=Sorted_eigenvalues_STE{1}(1);
    E2_STE{Neuron}=Sorted_eigenvalues_STE{1}(2);
    E1_Prior{Neuron}=Sorted_eigenvalues_Prior{1}(1);
    E2_Prior{Neuron}=Sorted_eigenvalues_Prior{1}(2);
    Eigen_Vector_Sorted_STE=eigen_vector_STE(:,Idenity_STE); %Rearrange eigen vectors
    Eigen_Vector_Sorted_Prior=eigen_vector_Prior(:,Idenity_Prior);
    E1_eigenvector_STE{Neuron}=Eigen_Vector_Sorted_STE(:,1);
    E2_eigenvector_STE{Neuron}=Eigen_Vector_Sorted_STE(:,2);
    E1_eigenvector_Prior{Neuron}=Eigen_Vector_Sorted_Prior(:,1);
    E2_eigenvector_Prior{Neuron}=Eigen_Vector_Sorted_Prior(:,2);
    clearvars  eigen_vector_STE eigen_value_STE eigen_vector_Prior...
        eigen_value_Prior Eigen_Vector_Sorted_STE Eigen_Vector_Sorted_Prior
    
    
    %% Plot Sorted Eigen Values and E1 and E2 for each neuron
    % Plot Eigenvaluesfor each neuron
    Num_Eigenvalues=50;
    Threshold=ones(1,Num_Eigenvalues);
    figure(7)
    %subplot(Num_Neurons,1,Neuron)
    plot(Sorted_eigenvalues_STE{Neuron}(1:Num_Eigenvalues),'ko') %plot first 50 eigenvalues
    hold on;
    plot(Sorted_eigenvalues_Prior{Neuron}(1:Num_Eigenvalues),'ro')
    plot(Threshold*E1_Prior{Neuron},'k--')
    xlabel('Sample(#)')
    ylabel('Eigen value')
    title(['Neuron ' num2str(Neuron)])
    legend('STE','Prior')
    hold off;
    
    savefig(figure(7),[FigPath,'/Signficant Eigenvalues/Significant Eigenvalues',Identity,'.fig'])
    %% 1D Nonlinear Decision Function
    % Calculate the non-linear decison function for each neuron
    
    for Spikes=1:Num_Spikes(Neuron)
        STA_Norm=STA(Neuron,:)/norm(STA(Neuron,:));%Jared Norm
        % Calculate P(s|spike)
        STE_Norm=STE_Store{Neuron}(Spikes,:)/norm(STE_Store{Neuron}(Spikes,:));
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
    figure(8)
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
    fit_x=Bin_Centers_Store{Neuron}(1):0.01:Bin_Centers_Store{Neuron}(end);
    NLD_Fit=spline(Bin_Centers_Store{Neuron},fire_rate{Neuron},fit_x);%polyval(pcoeffs_NLD,fit_x);
    NLD_Fit(end)=fire_rate{Neuron}(end);
    plot(fit_x,NLD_Fit,'r','LineWidth',5)
    hold off
    savefig(figure(8),[FigPath,'/NLDs/1D NLD',Identity,'.fig'])
    %% Predicted Spike Rate
    % Template=STA
    % Stimulus projection onto WN
    %% Choose Stimulus Type and Construct Spike Train
    % Choose Stimulus type
    for Sine_Struct_Num=1:3
        disp(num2str(Sine_Struct_Num))
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
        % This parameter determines the number of cycles to analyze (*2
        % equals 1 cycle, *4 equals 2 cycles, and so on and so forth)
        Stim_Template=Stim(1:end); %(1:1600*4) %For prediction 3 sine cycles start at index 1600
        Sine_Fig_Length=1:1600*90;
        %Stim=UpSample_WN_Stim(1:end-400);
        %t=linspace(0,length(WN_Stim)/1000,length(UpSample_WN_Stim));
        Fig_Count=0;
        % FIND Min Index
        %[V,idx]=min(Stim(1:1600));
        %% Predict spikes given a neurons STA and NLD
        
        for j=1:length(Stim_Template)
            %disp(num2str(j))
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
        
        % Determine how many sine repeats there are
        Sine_Window=length(Stim_Template);
        Num_Sine_Repeats=length(Stim(1:length(Stim_Template)))/Sine_Window; %1600 is 40ms sine presentation
        Sine_Phase_Stim=Stim(1:1600);
        
        %Construct Spike Train during Sine
        
        indice_count=0;
        
        for Sample_Num=1:Sine_Window;
            indice_count=indice_count+1;
            Sine_Sample_Num=Sine_Start_Indice(Sine_Struct_Num)+indice_count;
            if any(round(Neuron_Spike_Time_Samples(:,Neuron))==Sine_Sample_Num)
                Sine_Spike_Train{Neuron}(Sample_Num)=1;
            else
                Sine_Spike_Train{Neuron}(Sample_Num)=0;
            end
            
        end
        
        %% Predicted spike rate and actual spiking plot
        %Sine_Stim_Time=linspace(0,length(Sine_Fig_Length)/FS,length(Sine_Fig_Length));
        figure(9)
        subplot(3,1,Sine_Struct_Num)
        plot(2.*Stim(1:length(Sine_Fig_Length)),'k','LineWidth',2)
        hold on;
        
        % Plot Sorted Neural Data
        Sine_Spike_Idx=find(Sine_Spike_Train{Neuron}(Sine_Fig_Length)==1);
        %Sine_Spike_Idx=[1,100,1000,1500];
        Sorted_Spike_Amp=-10.*ones(1,length(Sine_Spike_Idx))-(5*Sine_Struct_Num);
        for j=1:length(Sine_Spike_Idx)
            plot([Sine_Spike_Idx(j), Sine_Spike_Idx(j)],[Sorted_Spike_Amp(j)-2.5, Sorted_Spike_Amp(j)+2.5],'k','LineWidth',1.5)
        end
        
        % Plot Predicted Spike Rate
        % Determine the offset
        Max_Predicted_Spike_Rate=max(Predicted_Fire_Rate_Proj{Neuron}(Sine_Fig_Length));
        Zero_Psr_Translate=Predicted_Fire_Rate_Proj{Neuron}(Sine_Fig_Length)-Max_Predicted_Spike_Rate;
        plot(0.5.*Zero_Psr_Translate-20-(5*Sine_Struct_Num),'b','LineWidth',2)
        
        title(['Predicted Spike Rate Sine ',num2str(Sine_Struct_Num),'V'])
        axis('tight')
        hold off
        % Save these results for reference because they are unmodified
        % because of plotting
        save([ResultPath,'/Base Unit Encoding Properties Results/Predicted Spike Rate and Sine Stim ',num2str(Sine_Struct_Num),'V ',Identity,'.mat'],...
            'Predicted_Fire_Rate_Proj','Stim','-mat')
        %% Xcorr the spike vector with the predicted spike vector
        [xcor,lag]=xcorr(Predicted_Fire_Rate_Proj{Neuron},Sine_Spike_Train{Neuron}(1:end-1600),'coeff');
        [~,I] = max(abs(xcor));
        lagDiff = lag(I);
        
        % Plot the Cross correlation
        figure(10)
        subplot(3,1,Sine_Struct_Num)
        plot(lag,xcor)
        ylabel('Cross Corr')
        xlabel('lag')
        axis('tight')
        title(['Cross Correlation Sine ',num2str(Sine_Struct_Num),'V'])
        
        
        %% Clear Variables
        clearvars Predicted_Fire_Rate_Proj Stim_Projection Stim_Norm STA_Norm Sine_Struct...
            Stim Stim_Template t xcor lag Zero_Psr_Translate Sine_Spike_Idx Sorted_Spike_Amp Sine_Spike_Train
    end
    savefig(figure(9),[FigPath,'/Sine Analysis/Spike Prediction',Identity,'.fig'])
    savefig(figure(10),[FigPath,'/Sine Analysis/Xcorr',Identity,'.fig'])
    
    
    
    
    
    %% Choose Stimulus Type and Construct Spike Train
    % Choose Stimulus type
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
        Fig_Count=0;
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
        
        
        %% Sine Raster
        Sine_Raster_Time=linspace(0,.040,1600);
        Sine_Raster_Stim=2*Sine_Phase_Stim+110;
        for Repeat=1:Num_Sine_Repeats
            figure(11+Sine_Struct_Num+Fig_Count)
            Ind_Spikes=find(Sine_Spike_Train{Neuron}(1:end,Repeat)==1);
            Raster_Row_Value=((Num_Sine_Repeats*ones(1,length(Ind_Spikes)))+2)-Repeat;
            %subplot(1,Num_Neurons,Neuron)
            plot(Sine_Raster_Stim,'k') %Plot WN stimulus over raster
            hold on;
            plot(Ind_Spikes,Raster_Row_Value,'k.');
            %hold on; %Uncomment to run raster without stimulus
            ylim([0, 130])
            xlim([0 length(Sine_Raster_Time)])
            xlabel('Sample Number')
            ylabel('Sine Phase Repeat#')
            title([Identity])
            clearvars Ind_Spikes Raster_Row_Value
        end
        
        savefig(figure(11+Sine_Struct_Num+Fig_Count),[FigPath,'/Sine Raster/Sine RasterV',num2str(Sine_Struct_Num),Identity,'.fig'])
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
        
        % Num_Spikes, mean, and std for each bin
        
        for Bin_Num=1:nbins
            Num_Spikes_Bin(Bin_Num)=sum(sum(Sine_Hist{Neuron,Bin_Num}));
            Mean_Spike_Bin(Bin_Num)=mean2(Sine_Hist{Neuron,Bin_Num});
            Std_Spike_Bin(Bin_Num)=std2(Sine_Hist{Neuron,Bin_Num});
        end
        
        % Plot Bar Graph of Actual Spike Rate
        figure(12+Sine_Struct_Num+Fig_Count)
        Bin_Centers=0:Sine_bin_index_width:Sine_Window-Sine_bin_index_width;
        Err_Center=Sine_bin_index_width/2:Sine_bin_index_width:Sine_Window;
        Actual_Spike_Rate=Num_Spikes_Bin./Bin_Time;
        
        % Construct Guassian window/ kernal
        guass_length=length(Actual_Spike_Rate);
        x_guass=1:guass_length;
        x0_guass=guass_length/2;
        A=1;
        W=6;
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
        
        bar(Bin_Centers,Actual_Spike_Rate,'histc','k')
        hold on;
        % Plot Convolved Actual Spike Rate
        plot(x_sampled,Smooth_Actual_Spike_Rate(Conv_SR),'r','LineWidth',2)
        % Predicted Spike Rate for Neuron
        plot(Predicted_Fire_Rate_Proj{Neuron},'b','LineWidth',2)
        %Stimulus
        plot(max(Num_Spikes_Bin./Bin_Time)+Sine_Phase_Stim+2,'k','LineWidth',2)
        xlabel('Sample Number(#)')
        ylabel('Spike Rate(Spikes/Sec)')
        legend('Actual Spike Rate','Covolved Actual Spike Rate','Predicted Spike Rate','Sine Cycle Stimulus')
        hold off;
        
        savefig(figure(12+Sine_Struct_Num+Fig_Count),[FigPath,'/Predicted & Actual Spike Rate/Pred and Actual SR SineV',num2str(Sine_Struct_Num),Identity,'.fig'])
        % figure(2)
        % bar(Bin_Centers,Mean_Spike_Bin,'histc','k')
        % errorbar(Err_Center,Mean_Spike_Bin,Std_Spike_Bin,'b')
        % % hold on;
        % % plot(max(Mean_Spike_Bin)+Sine_Phase_Stim+2,'k','LineWidth',2);
        % xlabel('Sample Number(#)')
        % ylabel('Mean Spike Count(#)')
        Fig_Count=Fig_Count+1;
        % %% Save Spike Trains during the Sinusoidal Stimulus
        % save('M16 Spike Train Sine.mat','Sine_Spike_Train','-mat')
        %% Clear Variables
        clearvars Sine_Hist Num_Spikes_Bin Mean_Spike_Bin Std_Spike_Bin bin_count...
            Predicted_Fire_Rate_Proj Stim_Projection Stim_Norm STA_Norm Sine_Struct Stim Stim_Template...
            t Smooth_Actual_Spike_Rate Actual_Spike_Rate y_guass x_guass y_cal Conv_SR x_sampled
    end
    clearvars -except neuron UpSample_WN_Stim WN_Stim Coh_Base_Units Base_Unit_Store; close all; clc
end








