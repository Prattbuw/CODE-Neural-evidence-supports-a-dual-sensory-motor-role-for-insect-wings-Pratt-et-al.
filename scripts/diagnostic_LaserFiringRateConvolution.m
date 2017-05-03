%------------------------------------
% plot diagnostic figure during analysis, can be toggled off in main run
% script
% TMOHREN 2016-12-21
% Last update 5/2/2017
%------------------------------------
f = figure(3);

set(f,'name','Firing rate convolution with gaussian')
f.Position = [600,600,400 400];
for q = 1:L.pulses(stimn)
     h(1:6) = fill([L.st(stimn,q);L.st(stimn,q);L.end(stimn,q);L.end(stimn,q)]/4e4,...
         [0,1,1,0]*max(F.drate(j,Pvec)),...
         [0.8,0.8,0.8],'EdgeColor','none');
        hold on
end
h(2) = plot(Pvec/4e4,F.drate(j,Pvec));
h(3) = plot(Pvec/4e4,F.crate(j,:));
h(4) = plot(Pvec/4e4,Gss/max(Gss)*18);
xlabel('Time [s]');ylabel('firing rate [spikes/sec]')
legend(h(1:4),'Laser ON','Discrete firing rate','Filtered firing rate','Gaussian filter')
hold off
drawnow