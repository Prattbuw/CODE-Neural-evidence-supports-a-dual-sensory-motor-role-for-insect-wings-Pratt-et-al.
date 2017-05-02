%% Analysis of Base and Tip Motion Stimuli
% NOTE: In this analysis, the whitenoise signal was not chunked up because
% for smoothing in the frequency domain because the resulting time
% domain signal was too sparse for later analyses 
% Created by Brandon Pratt
% Date: Janurary 2017
close all; clear all; clc

%% Load Data
% Load Digitzed M16 Data % Pt 1 and 2 are base pts
% Wing blade pt=11 and wing tip pt=4;
Total_Frames=11000;
[M, X_Rotated,Y_Rotated,Z_Rotated]=CoordinateTransform_V2('M16_Digitized_pts_1-17_Donexyzpts.csv',4,11,Total_Frames);

% Load Digitzed M28 Data % Pt 1 and 2 are base pts
% Wing blade pt=5 and wing tip pt=6;
Total_Frames_2=11010;
[M_2, X_Rotated_2,Y_Rotated_2,Z_Rotated_2]=CoordinateTransform_V2('M28_Pt_7xyzpts.csv',6,5,Total_Frames_2);

% Load Moth 27 digitized points
%Wing blade=5 and wing tip=6. wing base=3
[M_3, X_Rotated_3,Y_Rotated_3,Z_Rotated_3]=CoordinateTransform_V2('M27_Pt_6V3xyzpts.csv',6,5,Total_Frames);

%Load Moth 26 digitized points
%Wing blade=5 and wing tip=6. wing base=2
[M_4, X_Rotated_4,Y_Rotated_4,Z_Rotated_4]=CoordinateTransform_V2('M26_Pt_6_V2xyzpts.csv',6,5,Total_Frames);

% Load 1st WN Segment to ensure time alignment and frame at which the WN starts
% Compare motor and video signal at tip to determine actual start frame
load('Tip Motor Stim.mat');

%% Isolate Digitized Base and Tip Motion Stimuli
% Base Vertical Displacement of each video
% Z_Base for Video 1 M16
Z_Base_M16=Z_Rotated(:,7); %Pt 7 is the most proximal point to the base
Z_Base_Displacement_M16=Z_Base_M16-mean(Z_Base_M16);

%Z_Base for Video 2 M28
Z_Base_M28=Z_Rotated_2(:,7); %Pt 7 is the most proximal point to the base
Z_Base_Displacement_M28=Z_Base_M28-mean(Z_Base_M28);

%Z_Base for Video 3 M27
Z_Base_M27=Z_Rotated_3(:,3); %Pt 3 is the most proximal point to the base
Z_Base_Displacement_M27=Z_Base_M27-mean(Z_Base_M27);

%Z_Base for Video 3 M26
Z_Base_M26=Z_Rotated_4(:,2); %Pt 2 is the most proximal point to the base
Z_Base_Displacement_M26=Z_Base_M26-mean(Z_Base_M26);

% Data Structure for Z Base
Z_Base{1}=Z_Base_Displacement_M16(1011:end); %account for motor start up phase
Z_Base{2}=Z_Base_Displacement_M26(1011:end);
Z_Base{3}=Z_Base_Displacement_M27(1011:end);
Z_Base{4}=Z_Base_Displacement_M28(1011:end);

% Z_Base{1}=Z_Base_Displacement_M16; %account for motor start up phase
% Z_Base{2}=Z_Base_Displacement_M26;
% Z_Base{3}=Z_Base_Displacement_M27;
% Z_Base{4}=Z_Base_Displacement_M28;

% Tip Vertical Displacement of each video
% Z_Tip for Video 1 M16
Z_Tip_M16=Z_Rotated(:,17); %Pt 17 is the most proximal point to the tip
Z_Tip_Displacement_M16=Z_Tip_M16-mean(Z_Tip_M16);

%Z_Tip for Video 2 M28
Z_Tip_M28=Z_Rotated_2(:,6); %Pt 6 is the most proximal point to the tip
Z_Tip_Displacement_M28=Z_Tip_M28-mean(Z_Tip_M28);

%Z_Tip for Video 3 M27
Z_Tip_M27=Z_Rotated_3(:,6); %Pt 6 is the most proximal point to the tip
Z_Tip_Displacement_M27=Z_Tip_M27-mean(Z_Tip_M27);

%Z_Tip for Video 3 M26
Z_Tip_M26=Z_Rotated_4(:,6); %Pt 6 is the most proximal point to the tip
Z_Tip_Displacement_M26=Z_Tip_M26-mean(Z_Tip_M26);

% Data Structure for Z Tip
Z_Tip{1}=Z_Tip_Displacement_M16(1011:end); %account for motor start up phase
Z_Tip{2}=Z_Tip_Displacement_M26(1011:end);
Z_Tip{3}=Z_Tip_Displacement_M27(1011:end);
Z_Tip{4}=Z_Tip_Displacement_M28(1011:end);

% Z_Tip{1}=Z_Tip_Displacement_M16; %account for motor start up phase
% Z_Tip{2}=Z_Tip_Displacement_M26;
% Z_Tip{3}=Z_Tip_Displacement_M27;
% Z_Tip{4}=Z_Tip_Displacement_M28;
%% Upsample Video Displacements (Base and Tip)
% Cell Array is structures as follows: {1}=M16, {2}=M26, {3}=M27, {4}=M28
Vid_Cell_Array{1}='M16';
Vid_Cell_Array{2}='M26';
Vid_Cell_Array{3}='M27';
Vid_Cell_Array{4}='M28';
% Number of Digitized Videos
Num_Vids=length(Z_Tip);

% for Vid=1:Num_Vids
%     % UpSample Stimulus Segment by 40x
%     % Upsample to match the stimulus sampling with the spike timestamp sampling
%     Time_Stim=linspace(0,length(Z_Base{Vid})/1000,...
%     length(Z_Base{Vid}));
%     % Upsample Stimulus
%     Upsample_Time_Stim=linspace(0,length(Z_Base{Vid})/1000,...
%     length(Z_Base{Vid})*40);
%     % Upsample Base Signal
%     Z_Base{Vid}=spline(Time_Stim,Z_Base{Vid},Upsample_Time_Stim).';
%     % Upsample Tip Signal
%     Z_Tip{Vid}=spline(Time_Stim,Z_Tip{Vid},Upsample_Time_Stim).';
%     clearvars Time_Stim Upsample_Time_Stim
% end
    
%% Correlate the Motor Tip Signal with the digitized wing tip signal
% Downsample tip motor stim instead of upsampling digitized data
Tip_Motor_Stim=downsample(Tip_Motor_Stim,40);

for Vid=1:Num_Vids
    [xcor,lag]=xcorr(Tip_Motor_Stim,Z_Tip{Vid});
    [~,I] = max(abs(xcor));
    lagDiff = lag(I);
  
    % Correct for lag to align signal
    Aligned_Motor_Tip=Tip_Motor_Stim(lagDiff:end);
   
    % Now make the tip digitized and aligned motor signals the same size
    ReSized_Motor_Signal=Aligned_Motor_Tip(1:length(Z_Tip{Vid}));
    
    % Convert Motor Signal to mm instead of volts
    pcoeff=polyfit(ReSized_Motor_Signal,Z_Tip{Vid},1);
    slope=pcoeff(1);
    Best_Fit_Line=polyval(pcoeff,ReSized_Motor_Signal);
    MM_Motor_Signal{Vid}=slope.*ReSized_Motor_Signal;
 
    % plot each of the Vids Tip and mm_motor correlations
    figure(1)
    subplot(2,2,Vid)
    plot(ReSized_Motor_Signal,Z_Tip{Vid},'k.')
    hold on
    plot(ReSized_Motor_Signal,Best_Fit_Line,'r','LineWidth',5);
    title(Vid_Cell_Array{Vid})
    ylabel('Video Displacement(mm)')
    xlabel('Motor Displacement(V)')
    clearvars Best_Fit_Line ReSized_Motor_Signal Aligned_Motor_Tip xcor lag
end

%% Split the Video Tip and base and adjusted motor signals into parts and xcorr each chunk
Num_Chunks=1; % Adjust to split the signal into more parts for smoothing of FFT
%Num_Chunks=40; %Very smooth Gain and Phase, but maybe too high of
%resolution. Num_Chunks=10 (Good as well, like Num_Chunks=40);
for Vid=1:Num_Vids
    Length_Chunk=length(MM_Motor_Signal{Vid})/Num_Chunks;
    Signal_Start=1;
    Signal_End=Length_Chunk;
    % Store each signal chunk
    for Chunk=1:Num_Chunks
        % Motor Tip Signal Chunk
        Chunk_Store_Motor{Vid}{Chunk}=MM_Motor_Signal{Vid}(Signal_Start:...
            Signal_End);
        % Video Tip Signal Chunk 
        Chunk_Store_Video{Vid}{Chunk}=Z_Tip{Vid}(Signal_Start:...
            Signal_End);
        % Base Signal Chunk
        Chunk_Store_Base{Vid}{Chunk}=Z_Base{Vid}(Signal_Start:...
            Signal_End);
        % Adjust Signal Window for next chunk
        Signal_Start=Signal_Start+Length_Chunk;
        Signal_End=Signal_End+Length_Chunk;
    end  
end

%% XCorr Signal Chunks
% Set Xcorr range
% If sampling frequency is at 40kHz
%maxlag=10000;

%If Sampling frequency is at 1kHz
maxlag=100;
Lag_Lim=maxlag;
Img_Count=0;
for Vid=1:Num_Vids
    for Chunk=1:Num_Chunks
        % Xcorr signals
         [xcor,lag]=xcorr(Chunk_Store_Motor{Vid}{Chunk},Chunk_Store_Video{Vid}{Chunk},Lag_Lim);
         [~,I] = max(abs(xcor));
         lagDiff = lag(I);
         Lag_Store{Vid}(Chunk)=lagDiff;
         % plot Xcorr for each video
         Img_Count=Img_Count+1;
         figure(2)
         subplot(Num_Vids,Num_Chunks,Img_Count)
         plot(lag,xcor)
         ylabel('Cross Corr')
         xlabel('lag')
         title([Vid_Cell_Array{Vid},' Chunk ',num2str(Chunk)])
         
         clearvars xcor lag lagDiff
    end
end



%% Correlate Video Base Displacement with Motor Tip and Video Tip Signals

for Vid=1:Num_Vids
    % Xcorr motor tip signal and wing base signal
    [xcor_Motor,lag_Motor]=xcorr(MM_Motor_Signal{Vid},Z_Base{Vid},maxlag);
    [~,I_Motor] = max(abs(xcor_Motor));
    lagDiff_Motor(Vid) = lag_Motor(I_Motor);
    
    % Xcorr Video tip signal and wing base signal
    [xcor_Video,lag_Video]=xcorr(Z_Tip{Vid},Z_Base{Vid},maxlag);
    [~,I_Video] = max(abs(xcor_Video));
    lagDiff_Video(Vid) = lag_Video(I_Video);
    
    % Plot Cross Correlations
    figure(3)
    subplot(2,2,Vid)
    plot(lag_Motor,xcor_Motor,'k')
    hold on
    plot(lag_Video,xcor_Video,'b')
    title(Vid_Cell_Array{Vid})
    ylabel('Cross Correlation')
    xlabel('Lag')
    legend('xcor Base & Motor','xcor Base & Vid Tip')
    hold off
    clearvars xcor_Motor lag_Motor I_Motor xcor_Video lag_Video I_Video
end

%% FFT of Motor Tip and Base Signals
%FS=4e4; %Sampling frequency
FS=1e3;
dt=1/FS; %Time Steps
Img_Count=0;

for Vid=1:Num_Vids 
    for Chunk=1:Num_Chunks
    Length=length(Chunk_Store_Base{Vid}{Chunk});
    n=length(Chunk_Store_Base{4}{1});%2^nextpow2(Length);
    
    %FFT of Base
    Y_Base{Vid}(:,Chunk)=fft(Chunk_Store_Base{Vid}{Chunk},n); 
    
    figure(4)
    Img_Count=Img_Count+1;
    freq = 1/(dt*Length)*(0:Length);  %create the x-axis of frequencies in Hz
    freq_band= freq(find(freq<=250 & freq>=1));  % only plot 1 to 1500 Hz
    subplot(Num_Vids,Num_Chunks,Img_Count)
    plot(freq_band,abs(Y_Base{Vid}(1:length(freq_band),Chunk)))
    title([Vid_Cell_Array{Vid},' FFT Z Base Chunk ', num2str(Chunk)])
    xlabel('f (Hz)')
    ylabel('|Y Base(f)|')
    axis('tight')
   
    %FFT of Tip Signal
    Y_Motor_Tip{Vid}(:,Chunk)=fft(Chunk_Store_Motor{Vid}{Chunk},n); % Adjust to change tip signal
    figure(5)
    subplot(Num_Vids,Num_Chunks,Img_Count)
    plot(freq_band,abs(Y_Motor_Tip{Vid}(1:length(freq_band),Chunk)))
    title([Vid_Cell_Array{Vid},' FFT Motor Tip Chunk ', num2str(Chunk)])
    xlabel('f (Hz)')
    ylabel('|Motor Tip(f)|')
    axis('tight')
    
    % Phase angle of base FFT for each chunk
    Phase_Base{Vid}(:,Chunk)=angle(Y_Base{Vid}(:,Chunk));
    
    % Phase angle of Motor FFT for each chunk
    Phase_Motor{Vid}(:,Chunk)=angle(Y_Motor_Tip{Vid}(:,Chunk));
    
    
    end
    
    % Mean base fft signal
    Y_Base_Mean{Vid}=mean(Y_Base{Vid},2); %mean(abs(Y_Base{Vid}),2);
    
    % Mean Motor Tip fft signal
    Y_Motor_Tip_Mean{Vid}=mean(Y_Motor_Tip{Vid},2);%mean(abs(Y_Motor_Tip{Vid}),2);
    
    % Plot Mean base FFT
    figure(6)
    subplot(2,Num_Vids,Vid)
    plot(freq_band,abs(Y_Base_Mean{Vid}(1:length(freq_band))));
    xlabel('f (Hz)')
    ylabel('|Mean Y Base(f)|')
    title([Vid_Cell_Array{Vid},' Mean Base FFT'])
    axis('tight')
    
    %Plot Mean Motor FFT
    subplot(2,Num_Vids,Vid+Num_Vids)
    plot(freq_band,abs(Y_Motor_Tip_Mean{Vid}(1:length(freq_band))));
    xlabel('f (Hz)')
    ylabel('|Mean Y Motot(f)|')
    title([Vid_Cell_Array{Vid},' Mean Motor FFT'])
    axis('tight')
   
    % Gain
    Gain{Vid}=Y_Base_Mean{Vid}./Y_Motor_Tip_Mean{Vid}; % Base signal(Output)/Tip signal(Input)
    figure(7)
    subplot(2,2,Vid)
    plot(freq_band,log10(abs(Gain{Vid}(1:length(freq_band)))),'k');
    xlabel('f (Hz)')
    ylabel('log10 Gain (Base Displacement/Tip Displacement)')
    title([Vid_Cell_Array{Vid}, ' Gain'])
    axis('tight')
    
    % Mean Phase for Base FFT signal
    Phase_Base_Mean{Vid}=mean(Phase_Base{Vid},2);
    
    % Mean Phase for Motor FFT signal
    Phase_Motor_Mean{Vid}=mean(Phase_Motor{Vid},2);
    
    % Plot Phase
    Phase_Diff{Vid}=Phase_Base_Mean{Vid}-Phase_Motor_Mean{Vid};
    figure(8)
    subplot(2,2,Vid)
    plot(freq_band,Phase_Diff{Vid}(1:length(freq_band)),'k');
    xlabel('f (Hz)')
    ylabel('Phase (rads)')
    title([Vid_Cell_Array{Vid}, ' Phase'])
    axis('tight')
      
end

%% Mean of FFTs, Gains, and Phases of Males and Females and both
% Reshape cells
Female_Vids=[2,4];
Male_Vids=[1,3];
count=0;
% Reshape for males
for Male_Vid=Male_Vids
    count=count+1;
    Male_Mean_Base_FFT(:,count)=Y_Base_Mean{Male_Vid};
    Male_Gain(:,count)=Gain{Male_Vid};
    Male_Phase(:,count)=Phase_Diff{Male_Vid};
end

% Mean of Male FFT, Gain, and Phase
Mean_Male_Mean_Base_FFT=mean(Male_Mean_Base_FFT,2);
Mean_Male_Gain=mean(Male_Gain,2);
Mean_Male_Phase=mean(Male_Phase,2);

% Plots of the Male FFT, Gain, and Phase
figure(9)
subplot(1,3,1)
plot(freq_band,abs(Mean_Male_Mean_Base_FFT(1:length(freq_band))),'k');
xlabel('f (Hz)')
ylabel('|Mean Male Y Base(f)|')
title('Mean Male Base FFT')
axis('tight')

subplot(1,3,2)
plot(freq_band,abs(Mean_Male_Gain(1:length(freq_band))),'k');
xlabel('f (Hz)')
ylabel('Gain')
title('Mean Male Gain')
axis('tight')

subplot(1,3,3)
plot(freq_band,Mean_Male_Phase(1:length(freq_band)),'k');
xlabel('f (Hz)')
ylabel('Phase')
title('Mean Male Phase')
axis('tight')

count=0;
% Reshape for Females
for Female_Vid=Female_Vids
    count=count+1;
    Female_Mean_Base_FFT(:,count)=Y_Base_Mean{Female_Vid};
    Female_Gain(:,count)=Gain{Female_Vid};
    Female_Phase(:,count)=Phase_Diff{Female_Vid};
end

% Mean of Female FFT, Gain, and Phase
Mean_Female_Mean_Base_FFT=mean(Female_Mean_Base_FFT,2);
Mean_Female_Gain=mean(Female_Gain,2);
Mean_Female_Phase=mean(Female_Phase,2);

% Plots of the Female FFT, Gain, and Phase
figure(10)
subplot(1,3,1)
plot(freq_band,abs(Mean_Female_Mean_Base_FFT(1:length(freq_band))),'k');
xlabel('f (Hz)')
ylabel('|Mean Female Y Base(f)|')
title('Mean Female Base FFT')
axis('tight')

subplot(1,3,2)
plot(freq_band,abs(Mean_Female_Gain(1:length(freq_band))),'k');
xlabel('f (Hz)')
ylabel('Gain')
title('Mean Female Gain')
axis('tight')

subplot(1,3,3)
plot(freq_band,Mean_Female_Phase(1:length(freq_band)),'k');
xlabel('f (Hz)')
ylabel('Phase')
title('Mean Female Phase')
axis('tight')

% Reshape Regardless of Sex
for Vid=1:Num_Vids
    Reshaped_Mean_Base_FFT(:,Vid)=Y_Base_Mean{Vid};
    Reshaped_Gain(:,Vid)=Gain{Vid};
    Reshaped_Phase(:,Vid)=Phase_Diff{Vid};
    Reshaped_Motor(:,Vid)=Y_Motor_Tip_Mean{Vid};
    %Reshaped_Z_Base(:,Vid)=Z_Base{Vid}(1:length(Z_Base{1}));
end

% Mean of All Sexes FFT, Gain, and Phase
Mean_Mean_Base_FFT=mean(Reshaped_Mean_Base_FFT,2);
Mean_Gain=mean(Reshaped_Gain,2);
Mean_Phase=mean(Reshaped_Phase,2);
Mean_Y_Motor=mean(Reshaped_Motor,2);
%Mean_Z_Base=mean(Reshaped_Z_Base,2);

% Plots of All Sexes FFT, Gain, and Phase
figure(11)
subplot(1,3,1)
plot(freq_band,abs(Mean_Mean_Base_FFT(1:length(freq_band))),'k');
xlabel('f (Hz)')
ylabel('|Mean All Sexes Y Base(f)|')
title('Mean Base FFT All Sexes')
axis('tight')

subplot(1,3,2)
plot(freq_band,abs(Mean_Gain(1:length(freq_band))),'k');
xlabel('f (Hz)')
ylabel('Gain')
title('Mean Gain All Sexes')
axis('tight')

subplot(1,3,3)
plot(freq_band,Mean_Phase(1:length(freq_band)),'k');
xlabel('f (Hz)')
ylabel('Phase')
title('Mean Phase All Sexes')
axis('tight')

%% Check the cross-correlation between different stimulus chunk sizes
% Change segement length by changing ending indice and then take xcor with 
% entire motor signal
Start_Indice=1;
% If sampling rate is at 40kHz
%End_Indice=10000:10000:length(Z_Base{1}); %Adjust this parameter to chunk length growth

% If sampling rate is at 1kHz
End_Indice=1000:100:length(Z_Base{1}); %Adjust this parameter to chunk length growth

for Vid=1:Num_Vids
    %Xcor each chunk
    for Chunk_Length=1:length(End_Indice)
        [xcor,lag]=xcorr(MM_Motor_Signal{Vid},...
            Z_Base{Vid}(Start_Indice:End_Indice(Chunk_Length)));
        [~,I]=max(abs(xcor));
        lagDiff = lag(I);
        Time_Diff{Vid}(Chunk_Length)=abs(lagDiff)/FS;
        Time_Chunk{Vid}(Chunk_Length)=length(Start_Indice:End_Indice(Chunk_Length))/FS;
        clearvars xcor lag 
    end
    % plot base displacement time chunk by time of peak correlation for each video
    figure(12)
    plot(Time_Chunk{Vid},Time_Diff{Vid})
    xlabel('Duration of Base Signal Chunk(sec)')
    ylabel('\Delta t(sec)')
    title('Correlation Shift Motor Tip Base')
    legend(Vid_Cell_Array{1},Vid_Cell_Array{2},Vid_Cell_Array{3},Vid_Cell_Array{4})
    axis('tight')
    hold on
end
        
hold off

% Xcorr-check between Tip Digitized and Base Digitized
for Vid=1:Num_Vids
    %Xcor each chunk
    for Chunk_Length=1:length(End_Indice)
        [xcor,lag]=xcorr(Z_Tip{Vid},...
            Z_Base{Vid}(Start_Indice:End_Indice(Chunk_Length)));
        [~,I]=max(abs(xcor));
        lagDiff = lag(I);
        Time_Diff_Dig{Vid}(Chunk_Length)=abs(lagDiff)/FS;
        Time_Chunk_Dig{Vid}(Chunk_Length)=length(Start_Indice:End_Indice(Chunk_Length))/FS;
        clearvars xcor lag 
    end
    
    % plot base displacement time chunk by time of peak correlation for each video
    figure(13)
    plot(Time_Chunk_Dig{Vid},Time_Diff_Dig{Vid})
    xlabel('Duration of Base Signal Chunk(sec)')
    ylabel('\Delta t(sec)')
    title('Correlation Shift Digitized Tip Base')
    legend(Vid_Cell_Array{1},Vid_Cell_Array{2},Vid_Cell_Array{3},Vid_Cell_Array{4})
    axis('tight')
    hold on
end
        
hold off

% Xcorr check between motor Tip and changing size digitized tip
for Vid=1:Num_Vids
    %Xcor each chunk
    for Chunk_Length=1:length(End_Indice)
        [xcor,lag]=xcorr(MM_Motor_Signal{Vid},...
            Z_Tip{Vid}(Start_Indice:End_Indice(Chunk_Length)));
        [~,I]=max(abs(xcor));
        lagDiff = lag(I);
        Time_Diff_Tip{Vid}(Chunk_Length)=abs(lagDiff)/FS;
        Time_Chunk_Tip{Vid}(Chunk_Length)=length(Start_Indice:End_Indice(Chunk_Length))/FS;
        clearvars xcor lag 
    end
    % plot base displacement time chunk by time of peak correlation for each video
    figure(14)
    plot(Time_Chunk_Tip{Vid},Time_Diff_Tip{Vid})
    xlabel('Duration of Tip Signal Chunk(sec)')
    ylabel('\Delta t(sec)')
    title('Correlation Shift Motor Tip and Tip Digitized')
    legend(Vid_Cell_Array{1},Vid_Cell_Array{2},Vid_Cell_Array{3},Vid_Cell_Array{4})
    axis('tight')
    hold on
end
        
hold off

%% Transform motor signal into base signal using the mean gain from all sexes

Transformed_Motor_Y=Mean_Gain.*Mean_Y_Motor;

% Frequency to Time Domain % Approximate of Base signal given the motor
% signal
y_Transformed_Base=real(ifft(Transformed_Motor_Y,n));

% Upsample y_Transformed_Base based on Num_Chunks
Time_Org=linspace(0,10,length(y_Transformed_Base));
Time_Upsample=linspace(0,10,length(y_Transformed_Base)*Num_Chunks);
Upsample_y_Base=spline(Time_Org,y_Transformed_Base,Time_Upsample);

% Plot Time Domain Signal
figure(15)
plot(Time_Upsample,Upsample_y_Base)
ylabel('Vertical Displacement')
xlabel('Time(s)')
title('Base signal given motor transform')
axis('tight')

%% Save Reconstructed Base Stimulus
save('Generalized Base Displacement.mat','Upsample_y_Base','-mat')

%% Correlate Base Measured by Base Reconstructed

% Convert Motor Signal to mm instead of volts
for Vid=1:Num_Vids
    % Calculate best fit line
    pcoeff=polyfit([Upsample_y_Base(1:length(Z_Base{Vid}))].',Z_Base{Vid},1);
    Best_Fit_Line=polyval(pcoeff,Upsample_y_Base(1:length(Z_Base{Vid})));
    
    % Calculate R^2 value
    %Compute the residual values as a vector of signed numbers:
    yresid = Z_Base{Vid} - [Best_Fit_Line].';
    %Square the residuals and total them to obtain the residual sum of squares:
    SSresid = sum(yresid.^2);
    %Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1:
    SStotal = (length(Z_Base{Vid})-1) * var(Z_Base{Vid});
    %Compute R2 using the formula given in the introduction of this topic:
    % R2 predicts X % of variance in y
    rsq = 1 - (SSresid/SStotal);

    % plot each of the Vids Base measured and predicted values (mean
    % transform)
    figure(16)
    subplot(2,2,Vid)
    plot(Upsample_y_Base(1:length(Z_Base{Vid})),Z_Base{Vid},'k.')
    hold on
    plot(Upsample_y_Base(1:length(Z_Base{Vid})),Best_Fit_Line,'r','LineWidth',5);
    title([Vid_Cell_Array{Vid},' Base Measured correlated to Base Predicted'])
    ylabel('Base Vertical Displacement Measured(mm)')
    xlabel('Base Vertical Displacement Predicted(mm)')
    txt1 = ['r^2 = ',num2str(rsq)];
    x1=max(Upsample_y_Base(1:length(Z_Base{Vid})))/2;
    y1=min(Z_Base{Vid})/3;
    text(x1,y1,txt1)
    
    clearvars yresid Best_Fit_Line pcoeff
end
 
%% Confidence Check with Transform

% Use Gain function to transform the motor signal for each moth
for Vid=1:Num_Vids
    
    Base_Pred=Gain{Vid}.*Y_Motor_Tip_Mean{Vid};
    y_Base_Pred=real(ifft(Base_Pred,n));

    % Upsample y_Transformed_Base based on Num_Chunks
    Time_Org=linspace(0,10,length(y_Base_Pred));
    Time_Upsample=linspace(0,10,length(y_Base_Pred)*Num_Chunks);
    Upsample_Base_Pred=spline(Time_Org,y_Base_Pred,Time_Upsample);
    
     % Calculate best fit line
    pcoeff=polyfit([Upsample_Base_Pred(1:length(Z_Base{Vid}))].',Z_Base{Vid},1);
    Best_Fit_Line=polyval(pcoeff,Upsample_Base_Pred(1:length(Z_Base{Vid})));
    
    % Calculate R^2 value
    %Compute the residual values as a vector of signed numbers:
    yresid = Z_Base{Vid} - [Best_Fit_Line].';
    %Square the residuals and total them to obtain the residual sum of squares:
    SSresid = sum(yresid.^2);
    %Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1:
    SStotal = (length(Z_Base{Vid})-1) * var(Z_Base{Vid});
    %Compute R2 using the formula given in the introduction of this topic:
    % R2 predicts X % of variance in y
    rsq = 1 - (SSresid/SStotal);

    % plot each of the Vids Base measured and predicted values for each
    % moth
    figure(17)
    subplot(2,2,Vid)
    plot(Upsample_Base_Pred(1:length(Z_Base{Vid})),Z_Base{Vid},'k.','MarkerSize',10)
    hold on
    plot(Upsample_Base_Pred(1:length(Z_Base{Vid})),Best_Fit_Line,'r','LineWidth',1);
    title([Vid_Cell_Array{Vid},' Check: Recovery of Base Displacement'])
    ylabel('Base Vertical Displacement Measured(mm)')
    xlabel('Base Vertical Displacement Predicted(mm)')
    txt1 = ['r^2 = ',num2str(rsq)];
    x1=max(Upsample_Base_Pred(1:length(Z_Base{Vid})))/2;
    y1=min(Z_Base{Vid})/3;
    text(x1,y1,txt1)
    
    clearvars yresid Best_Fit_Line pcoeff Upsample_Base_Pred y_Base_Pred Upsample_Base_Pred...
        Time_Upsample Time_Org
end
    
    
    
    
