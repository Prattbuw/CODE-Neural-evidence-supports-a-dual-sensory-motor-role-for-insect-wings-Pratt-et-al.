%% Motor Voltage and Position Calibration
% Created by Brandon Pratt
% Date: October 2016
clear all; close all; clc
% Average of 5 measurements at each voltage
Volt=[0 .5980 .9980 1.1970 1.5960 1.9940 2.3940 2.7910 3.1920 3.5880 3.9886 4.3870 4.7878 4.6681 4.8641 5.0616 5.2864]; %Remeasure
log_Volt=log10(Volt);
% Sapcing of oculometer scale
Spacing=[0 11 18 21 28 36 43 50 55 62 69 76 81 80 84 86 89];

% Conversion of spacing into mms using calipers
Space_mm_Conversion=5/42; %mm/spaces %Note 84 measures spaces equals 10mm
mm_Motor_Displacement=Spacing.*Space_mm_Conversion;
log_mm_Motor_Displacement=log10(mm_Motor_Displacement);
%Calulate Coefficents
p=polyfit(Volt,Spacing,3);
p2=polyfit(Volt,mm_Motor_Displacement,3);
p3=polyfit(log_Volt(2:end),log_mm_Motor_Displacement(2:end),1);
p4=polyfit(Volt,mm_Motor_Displacement,2);
Y=polyval(p,Volt);
Y2=polyval(p2,Volt);
Y3=polyval(p3,log_Volt);
Y4=polyval(p4,Volt);
%% Calculate R^2 value
%Compute the residual values as a vector of signed numbers:
yresid = mm_Motor_Displacement - Y2;
%Square the residuals and total them to obtain the residual sum of squares:
SSresid = sum(yresid.^2);
%Compute the total sum of squares of y by multiplying the variance of y by the number of observations minus 1:
SStotal = (length(mm_Motor_Displacement)-1) * var(mm_Motor_Displacement);
%Compute R2 using the formula given in the introduction of this topic:
% R2 predicts X % of variance in y
rsq = 1 - (SSresid/SStotal)


%% Plot Fits
figure(1)
% subplot(1,3,1)
% plot(Volt,Spacing,'k.',Volt,Y,'r');
% xlabel('Voltage Input(V)')
% ylabel('Ocular Micrometer Spcacing Output')
subplot(1,2,1)
plot(Volt,mm_Motor_Displacement,'k.',Volt,Y4,'r');
xlabel('Voltage Input(V)')
ylabel('Motor Tip Position Output(mm)')
title('Quadratic Fit')

subplot(1,2,2)
plot(Volt,mm_Motor_Displacement,'k.',Volt,Y2,'r');
xlabel('Voltage Input(V)')
ylabel('Motor Tip Position Output(mm)')
txt1 = '\leftarrow R^2 = 0.9983';
x1=max(Volt)/2;
y1=max(mm_Motor_Displacement)/2;
text(x1,y1,txt1)
title('Cubic Fit')

figure(2)
plot(log_Volt,log_mm_Motor_Displacement,'k.',log_Volt,Y3,'r');
xlabel('Log Voltage Input(V)')
ylabel('Log Motor Tip Position Output(mm)')

%% Conversion Function of Volts to MM
MM_log_linear=@(V) p3(1).*V+p3(2);
MM_Quadratic=@(V) p4(1).*V.^2 +p4(2).*V + p4(3);
MM_Cubic=@(V) p2(1).*V.^3 + p2(2).*V.^2+p2(3).*V+p2(4);
%% Load White Noise Segment and Video Data
White_Noise_Segment=load('Whitenoise_Repeat1.txt');
Total_Frames=11000;
[M, X_Rotated,Y_Rotated,Z_Rotated]=CoordinateTransform_V2('M16_Digitized_pts_1-17_Donexyzpts.csv',4,11,Total_Frames);
Actual_Motor_Output=load('Actual_Motor_Ouput.txt');

%% Transform Wing Tip Vector
X_Tip=X_Rotated(:,end);
Y_Tip=Y_Rotated(:,end);
Z_Tip=Z_Rotated(:,end);

% Calulate the Initial Position of the Wing
X0=X_Tip(1);
Y0=Y_Tip(1);
Z0=Z_Tip(1);

%Intitial Position in terms of mean (Use this as it accounts for varability
X0_mean=mean(X_Tip);
Y0_mean=mean(Y_Tip);
Z0_mean=mean(Z_Tip);

% Repeat vector of initital Positions
Rep_X0_mean=repmat(X0_mean,length(X_Tip),1);
Rep_Y0_mean=repmat(Y0_mean,length(Y_Tip),1);
Rep_Z0_mean=repmat(Z0_mean,length(Z_Tip),1);

% Calculate change in position per frame;
Delta_X=X_Tip-Rep_X0_mean;
Delta_Y=Y_Tip-Rep_Y0_mean;
Delta_Z=Z_Tip-Rep_Z0_mean;

% Calculate Magnitude of Position vector D
D=sqrt(Delta_X.^2+Delta_Y.^2+Delta_Z.^2);
Neg_Wing=Z_Tip<Z0_mean;
Neg_Wing_Z=Neg_Wing.*-1;
D_Neg_Sign=D.*Neg_Wing_Z;
Pos_Wing=Z_Tip>Z0_mean;
D_Pos_Sign=D.*Pos_Wing;
D_Correct_Sign=D_Pos_Sign+D_Neg_Sign;

%% Plot Transform
subplot(3,1,1)
plot(D_Correct_Sign)
ylabel('Tip Diplacement(mm)')
title('Tip Vector')

subplot(3,1,2)
plot(Z_Tip-mean(Z_Tip))
ylabel('Tip Diplacement(mm)')
title('Digitized Z')

subplot(3,1,3)
plot(Z_Tip-mean(Z_Tip),D_Correct_Sign,'bo')
ylabel('Tip Diplacement(mm)')
title('Digitized Z vs Tip Vector')
%% Align WN segment with Video
% Tip Point 
frame_length=11000;
Time=linspace(0,10,frame_length);
Motor_Output=White_Noise_Segment;
% Down Sample Motor Ouput Voltage
Down_Sample_Actual_Motor_Output=downsample(Actual_Motor_Output(1:440000),40);
% Down Sample Motor input Voltage
Down_Sample_Motor_Output=downsample(White_Noise_Segment,40);
%Input Motor Voltage
Motor_Output_Segment=Down_Sample_Motor_Output;  %First 1s
%Output Motor Voltage
Actual_Motor_Output_Segment=Down_Sample_Actual_Motor_Output-mean(Down_Sample_Actual_Motor_Output);
%Video Segment
Video_Segment=D_Correct_Sign(1:frame_length); %First 1s
% Convert Motor Voltage Into mm
MM_log_Motor=MM_log_linear(Actual_Motor_Output_Segment);
% Correct for log
MM_Motor=10.^MM_log_Motor;

MM_Quad_Motor=MM_Quadratic(Actual_Motor_Output_Segment);
MM_Cubic_Motor=MM_Cubic(Actual_Motor_Output_Segment);

% %% Xcorr input and output motor voltage
% Fs=1000;
% [Motor_IO_Correlation, Motor_lag]=xcorr(Motor_Output_Segment,Actual_Motor_Output_Segment);
% [~,I_Motor] = max(abs(Motor_IO_Correlation));
% LagDiff = Motor_lag(I_Motor);
% TimeDiff = LagDiff/Fs;
% 
% plot(Motor_lag,Motor_IO_Correlation)
% Zero_Padding_Motor=zeros(1,abs(lagDiff)).';
% 
% Motor_Zero_Pad=[Zero_Padding_Motor;Motor_Output_Segment];
% MotorI_Align=Video_Zero_Pad(1:end-length(Zero_Padding_Motor));
% MotorO_Align=Actual_Motor_Output_Segment;
% 
% %% Plot Motor Input and Output Voltages
% plot(MotorI_Align-mean(MotorI_Align),'r')
% hold on;
% plot(MotorO_Align,'k')
% ylabel('Motor Voltage(V)')
% legend('Motor Input Voltage', 'Motor Output Voltage')
%% Plot Video and Motor Segments
subplot(4,1,1)
plot(Time,Video_Segment);
ylabel('Displacement(mm)')
xlabel('Time(s)')
title('Video Tip Displacement')

subplot(4,1,2)
plot(Time, MM_Motor)
ylabel('Displacement(mm)')
xlabel('Time(s)')
title('Motor Linear Via Log')

subplot(4,1,3)
plot(Time, MM_Quad_Motor)
ylabel('Displacement(mm)')
xlabel('Time(s)')
title('Motor Quadratic')

subplot(4,1,4)
plot(Time, MM_Cubic_Motor)
ylabel('Displacement(mm)')
xlabel('Time(s)')
title('Motor Cubic')
%% Correlate the Motor Output and Video
Fs=1000;
[Motor_Video_Correlation, lag]=xcorr(Video_Segment,MM_Quad_Motor);
[~,I] = max(abs(Motor_Video_Correlation));
lagDiff = lag(I);
timeDiff = lagDiff/Fs;

plot(lag,Motor_Video_Correlation)
Zero_Padding=zeros(1,abs(lagDiff)).';
%% Align Motor and Video
Video_Zero_Pad=[Zero_Padding;Video_Segment];
Video_Align=Video_Zero_Pad(1:end-length(Zero_Padding));
Motor_Align=MM_Cubic_Motor;
Time_Align=linspace(0,11,length(Video_Align));
%% Plot Alignment
subplot(3,1,1)
plot(Time_Align,Motor_Align)
xlabel('Time(sec)')
ylabel('Tip Displacement(mm)')
title('Motor Align')

subplot(3,1,2)
plot(Time_Align,Video_Align)
xlabel('Time(sec)')
ylabel('Tip Displacement(mm)')
title('Video Align')

subplot(3,1,3)
plot(Time_Align,Motor_Align,'b',Time_Align,Video_Align,'r')
xlabel('Time(sec)')
ylabel('Tip Displacement(mm)')
title('Video and Motor Align Overlay')
legend('Motor Align','Video Align')
%% Conversion Factor
% Displacement = abs(Motor_Align./Video_Align); %Correct Conversion factor at each index
coeffs = polyfit(Motor_Align,Video_Align, 1);
% Now get the slope, which is the first coefficient in the array:
slope = coeffs(1)
y=slope.*Motor_Align+coeffs(2); % fitted line

%% Plot Conversion
figure(1)
plot(Motor_Align,Video_Align,'b',Motor_Align,y,'r');
xlabel('Motor Displacement(V)')
ylabel('Video Displacement(mm)')

figure(2)
subplot(3,1,1)
plot(Time_Align,Video_Align)
title('Video Align')
ylabel('Displacement(mm)')
xlabel('Time(ms)')

subplot(3,1,2)
plot(Time_Align,Motor_Align.*slope) %Motor is Aligned
title('Motor Align')
ylabel('Displacement(mm)')
xlabel('Time(ms)')

subplot(3,1,3)
plot(Time_Align,Motor_Align.*slope,'b',Time_Align,Video_Align,'r')
xlabel('Time(sec)')
ylabel('Tip Displacement(mm)')
title('Video and Motor Align Overlay')
legend('Motor Align','Video Align')