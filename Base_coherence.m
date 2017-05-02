% Analyze Gaussian white noise signal to show signal-response coherence
%   for a single neuron
% Created by Brandon Pratt
% Date August 2016
clear all; close all; clc;
%% Load Data
% Stimulus Data
load('UpSampled_Generalized Base Displacement.mat') % Reconstructed Base Stimulus (after upsampling)
UpSample_WN_Stim=Upsample_y_Base;
%Spike Trains
load('Wing_Base_Identity_Spike_Trains.mat')
Num_Neurons=length(Base_Unit_Store(1,1:end));
%% Run Analysis for all base idenitfied neurons
for neuron=2:Num_Neurons %Loop for permutating all neurons
%neuron = 3; %Neuron for loop this with data strucutre
Identity=Base_Unit_Store{1,neuron}; %Moth number and neuron number identity data structure
WN_Repeat_Matrix=Base_Unit_Store{2,neuron}; %Spike train for particular neuron
disp(Identity) %disp Identity
fs = 4e4;    % In kHz

%% Plot PSD of white noise signal
display('Calculating Power Spectrum Density')
stim_length = 10*fs;       % In samples
t = (1:stim_length)/fs;
stim_block = UpSample_WN_Stim(1:end);

tau_max = 1e4;  % Time to calculate autocorrelation in ms
tau = -tau_max:(1e3/fs):tau_max;

R = xcorr(stim_block, (numel(tau)-1)/2, 'coeff');  % Since mean=0 and SD=1

% Plot autocorrelation of signal
figure
plot(tau, R)
xlabel('Time \tau (sec)')
ylabel('BLGWN autocorrelation R(\tau)')

% PSD estimate is FT of autocorrelation
N = numel(R);
f = fs*(0:(N/2))/N;         % Freq in Hz
P = (2/N)*abs(fft(R)).^2;
P = P(1:numel(f));

figure('Name', 'Power Spectrum Density')
plot(f, 10*log10(P))
xlabel('Frequency (Hz)')
ylabel('Power (dB)')
xlim([0 300])

%% Convolve spike train to add width
Num_Repeats=30;
Gauss_Width=gaussmf(-20:20,[5, 0]);
for j=1:Num_Repeats
    Spikes_Width(:,j)=conv(WN_Repeat_Matrix(:,j),Gauss_Width,'same');
end
%% Plot Spikes during a segment of whitenoise
% time=(1:10000)./4e4;
% subplot(2,1,1)
% plot(time,spikes(1,1:10000));
% subplot(2,1,2)
% plot(time,Spikes_Width(1,1:10000));
%% SR Coherence
figure, hold on
f = 1:250; %frequency range
disp(['Coherence for ', Identity])
C_rs = zeros(size(WN_Repeat_Matrix(:,:),2), numel(f));
for i=1:size(WN_Repeat_Matrix(:,:),2)
    display(sprintf('    Repeat %d', i))
    window = fs;
    noverlap = fs/2;
    [C_rs(i, :), F] = mscohere(stim_block, Spikes_Width(:,i), window, noverlap, f, fs); %can use spikes instead of Spikes_Width
    plot(F, C_rs(i, :))
end

%% Plot means
avg_C = mean(C_rs, 1);
err_C = std( C_rs, 0, 1 );
figure, hold on

h =    fill(    [F, fliplr(F)], [avg_C-err_C...       
                fliplr(avg_C+err_C)],...
        [0.8,0,0], 'EdgeColor','none');   % %             [0.9,0.9,1], 'EdgeColor','none');  
set(h,'facealpha',.2);

plot(F, squeeze(avg_C  ),'r')
ylim([0 0.7])
ylabel('C_{RS}')
xlabel('f (Hz)')

%% Permutations of spike trains
Num_Perms=40;
clear j;
for j=1:Num_Repeats
    for k=1:Num_Perms
        Perm_Indices=randperm(length(WN_Repeat_Matrix(:,j)));
        spikes_segment=WN_Repeat_Matrix(:,j);
        Perm_Spikes(k,:)=spikes_segment(Perm_Indices);
    end
    Permutated_Spikes(:,:,j)=Perm_Spikes;
end
%% Convolve Permutated_Spikes with guassin kernal
Gauss_Width=gaussmf(-20:20,[5, 0]);%add width to the spikes of 1ms Gauss
clear j;
clear k;
Whitenoise_Repeats=27;
for j=1:Num_Repeats
    for k=1:Num_Perms
        Conv_Perm_Spikes(k,:,j)=conv(Permutated_Spikes(k,:,j),Gauss_Width,'same');
    end
end

%% Plot Permutated
% time=(1:10000)./4e4;
% subplot(4,1,1)
% plot(time,Permutated_Spikes(1,1:10000,1));
% subplot(4,1,2)
% plot(time,Conv_Perm_Spikes(1,1:10000,1));
% subplot(4,1,3)
% plot(time,Permutated_Spikes(2,1:10000,1));
% subplot(4,1,4)
% plot(time,Conv_Perm_Spikes(2,1:10000,1));

%% SR Coherence of Permutated Convoled Spikes
% figure, hold on
f = 1:250; %frequency range
Perm_SR_Coherence = zeros(40, numel(f), Num_Repeats);
window = fs;
noverlap = fs/2;
clear j;
clear k;

for j=1:Num_Repeats
    display(sprintf('j=%d',j))
    for k=1:Num_Perms
        [Perm_SR_Coherence(k,:,j),F]= mscohere(stim_block, Conv_Perm_Spikes(k,:,j) , window, noverlap, f, fs);
        display(sprintf('k=%d',k))
    end
end

%% Reshape Perm_SR_Coherence
Reshaped_Perm_SR_Coherence=zeros(Num_Perms*Num_Repeats,numel(f));
clear j;
clear k;
ind=0:40:(Num_Perms*Num_Repeats)-40;
for j=1:Num_Repeats % 30 repeats of whitenoise used
    for k=1:Num_Perms
    indice=k+ind(j);
    Reshaped_Perm_SR_Coherence(indice,:)= Perm_SR_Coherence(k,:,j);  
    end 
end

%% Sort Reshape_Perm_SR_Coherence from largest to smallest
% Identity: moth and neuron number string
Sorted_Reshape_Perm_SR_Coherence=sort(Reshaped_Perm_SR_Coherence,1);
ResultsPath='C:\Users\Daniellab\Documents\Brandon Wing Data\Base Coherence Analysis';
save([ResultsPath,'\Coherence Results\Sorted Reshape Perm SR Coherence neuron ',Identity,'.mat'],'Sorted_Reshape_Perm_SR_Coherence','-mat')
% save(['Sorted Reshape Perm SR Coherence neuron 1.mat'],'Sorted_Reshape_Perm_SR_Coherence','-mat')
%% Mean, error, and 95% confidence interval
avg_Perm_SR_Coherence= mean(Sorted_Reshape_Perm_SR_Coherence, 1);
err_Perm_SR_Coherence= std( Sorted_Reshape_Perm_SR_Coherence, 0, 1 );
%95 confidence interval
Indices_Removal=.05*1080; % 2.5% of the indices to be removed from each side
Percent95_Confidence_Interv=Sorted_Reshape_Perm_SR_Coherence(1:end-Indices_Removal,:);
Percent95_Width=(Percent95_Confidence_Interv(end,:)-avg_Perm_SR_Coherence);
%Determine if the coherence is outside of the 95% interval at each frequency
Percent95_Confidence_Max_Value=Percent95_Confidence_Interv(end,:);
Significant_Coherence_Values_std=(avg_C-err_C)>Percent95_Confidence_Max_Value; %Use for a more conservative measure of significance
Significant_Coherence_Values=avg_C>Percent95_Confidence_Max_Value;
%% Plot Mean, error, and 95% confidence interval
figure(10), hold on
%Plot the freqencies that have the significant coherence values
Plot_Sig_Coherence=[Significant_Coherence_Values]+0;
Plot_Sig_Coherence(Plot_Sig_Coherence==0)=NaN;
%plot the freqencies that have the significant coherence values on the
%curve itself
indices_of_Sig_Coherence=find(Plot_Sig_Coherence==1);

% Plot a magenta bar at frequencies that have a signifcant coherence value
% bar(Plot_Sig_Coherence,1,'m','EdgeColor','none')

h =    fill(    [F, fliplr(F)], [avg_Perm_SR_Coherence-Percent95_Width...       
                fliplr(avg_Perm_SR_Coherence+Percent95_Width)],...
        [0.8,0,0], 'EdgeColor','none');   % %             [0.9,0.9,1], 'EdgeColor','none');  
set(h,'facealpha',.2);

plot(F, squeeze(avg_Perm_SR_Coherence),'r')
ylim([0 0.7])
ylabel('Coherence_{SR}')
xlabel('Frequency(Hz)')
std_plot=    fill(    [F, fliplr(F)], [avg_C-err_C...       
                fliplr(avg_C+err_C)],...
        [0,0,0.8], 'EdgeColor','none');   % %             [0.9,0.9,1], 'EdgeColor','none');  
set(std_plot,'facealpha',0.2);


avg_C = mean(C_rs, 1);
err_C = std( C_rs, 0, 1 );
plot(F, squeeze(avg_C ),'b')
% legend('Significant Coherence Frequency','95% Confidence Interval','Average Perm SR Coherence','Average Coherence STD','Average Coherence');
legend('95% Confidence Interval','Average Perm SR Coherence','Average Coherence STD','Average Coherence');
plot(Plot_Sig_Coherence.*0.5,'k*')
% ylabel('Coherence Outside of the 95% Confidence Interval')
%Plotting Circles
plot(indices_of_Sig_Coherence,avg_C(indices_of_Sig_Coherence),'mo')

% Save Figure
FigPath='C:\Users\Daniellab\Documents\Brandon Wing Data\Base Coherence Analysis';
savefig(figure(10),[FigPath,'\Coherence Figs\SR Coh ',Identity,'.fig'])
%% Save Permutated Mean, std, and 95% confidence interval
%Identity: moth and unit number string
save([ResultsPath,'\Coherence Results\Perm Mean, std, 95% Confidence, Frequencies of Sig Coh, Significant_Coherence_Values, and Neuron ',Identity,'.mat'],...
    'avg_Perm_SR_Coherence','err_Perm_SR_Coherence','Percent95_Confidence_Interv',...
    'indices_of_Sig_Coherence','Significant_Coherence_Values','-mat');
% save(['Perm Mean, std, 95% Confidence, Significant_Coherence_Values, and Neuron 1.mat'],'avg_Perm_SR_Coherence','err_Perm_SR_Coherence','Percent95_Confidence_Interv','Significant_Coherence_Values','-mat');
% clearvars -except NEURON 
% end

%% Distrubution of Permutation Coherences for a Particular Frequency
hold off
Frequency_Distrubution=Sorted_Reshape_Perm_SR_Coherence(:,1);%Choose Frequency
nbins=15;
Frequency_Distrubution_Histogram=histogram(Frequency_Distrubution,nbins);
xlabel('SR Coherence')
ylabel('Count')
hold on
Count_Values=Frequency_Distrubution_Histogram.Values;
Bin_Length=Frequency_Distrubution_Histogram.BinWidth/2:Frequency_Distrubution_Histogram.BinWidth:Frequency_Distrubution_Histogram.BinLimits(2)-Frequency_Distrubution_Histogram.BinWidth/2;
plot(Bin_Length,Count_Values,'mo-');
hold off
%% Total Information Rate (Bits per Second)
% Total Information rate
df = F(2)-F(1);
Total_Information_Function= -log2((1-avg_C)*df);
Integrate_Total_Information_Function=trapz(Total_Information_Function);
disp(['Total Information Rate= ',int2str(Integrate_Total_Information_Function),'(Bits/s)']);
% approx_bit_rate = df*sum(-log2(1-avg_C))
%% Signifcant Band Information Rate
% for j=2:length(indices_of_Sig_Coherence)
%     if indices_of_Sig_Coherence(j)-indices_of_Sig_Coherence(j-1)==1
%         Band_Indices(j-1)=indices_of_Sig_Coherence(j-1);
%         Band_Indices(j)=indices_of_Sig_Coherence(j);
%     end
% end
% Non_Zero_Band_Indices=find(Band_Indices>0);
% Coherence_Band_Indices=Band_Indices(Non_Zero_Band_Indices);%Bands of Frequencies with signifcant coherences

% %% Calculate Band Coherence Indices
% % Determine the Indicess of the frequency bands that have signifcant Coherence
% clear j;
% Band_Matrix(1,1)=1;
% C=1;
% for j=2:length(Coherence_Band_Indices)
%     if Coherence_Band_Indices(j)-Coherence_Band_Indices(j-1)~=1
%         Band_Matrix(C,2)=j-1;
%         C=C+1;
%         Band_Matrix(C,1)=j;
%     end
% end
% Band_Matrix(end,2)=length(Coherence_Band_Indices);

% %% Calculate the Information Rate of Each Frequency Band
% clear j;
% df = F(2)-F(1);
% for j=1:length(Band_Matrix(:,1))
%     Frequency_Band=Band_Matrix(j,:);
%     Band_Information_Function=-log2((1-avg_C(Coherence_Band_Indices(Band_Matrix(j,1)):Coherence_Band_Indices(Band_Matrix(j,2))))*df);
%     Frequency_Band_Information_Rate(j)=trapz(Band_Information_Function);
%     disp(['Frequency Band ',num2str(Coherence_Band_Indices(Band_Matrix(j,1))),'Hz To ',num2str(Coherence_Band_Indices(Band_Matrix(j,2))),'Hz Information Rate= ',num2str(Frequency_Band_Information_Rate(j)),'(Bits/s)']);
% end
 
%% Calculate Overall Spike Rate (Spikes per Second)
Total_Time=Num_Repeats.*10; %Seconds
Total_Number_Spikes=sum(sum(WN_Repeat_Matrix(:,:)));
Spike_Rate=Total_Number_Spikes./Total_Time;

%% Calculate Information Per Spike (Bits/Spike)
Information_Per_Spike=Integrate_Total_Information_Function./Spike_Rate;
disp(['Information Per Spike= ',num2str(Information_Per_Spike),'(Bits/Spike)']);

%% Save Total Information Rate, Spike Rate, and Information Per Spike for each neuron
%With Identity alter moth and neuron string
save(([ResultsPath,'\Coherence Results\Spike Rate, Info Rate, and Info Per Spike neuron ',Identity,'.mat']),'Spike_Rate','Integrate_Total_Information_Function','Information_Per_Spike','-mat');
%% Calculate the Information Per Spike in Each Signifcant Band Frequency
% Band_Information_Per_Spike=(Frequency_Band_Information_Rate./Integrate_Total_Information_Function).*Information_Per_Spike;
% 
% % Display Information Per Spike For Each Band Frequency
% clear j;
% for j=1:length(Band_Information_Per_Spike)
%     disp(['Frequency Band ',num2str(Coherence_Band_Indices(Band_Matrix(j,1))),'Hz To ',num2str(Coherence_Band_Indices(Band_Matrix(j,2))),'Hz Information Per Spike= ',num2str(Band_Information_Per_Spike(j)),'(Bits/Spike)']);
% end
clearvars -except UpSample_WN_Stim neuron Num_Neurons Base_Unit_Store; close all; clc;
end









