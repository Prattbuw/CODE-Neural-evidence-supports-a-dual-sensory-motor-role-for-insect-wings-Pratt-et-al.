%------------------------------------
% diagnostic fig, can be toggled off in main script
% TMOHREN 2016-12-21
% Last update 5/2/2017
%------------------------------------
f = figure(1);
set(f,'name','Laser start & stop')
f.Position = [300,200,700 600];

 for k = 1:L.pulses(j)
 h(1) = fill([L.st(j,k);L.st(j,k);L.end(j,k);L.end(j,k)]/40e3,...
     [0,1,1,0]*3.4,...
     [0.9,0.9,0.9],'EdgeColor','none');
    hold on
 end
h(2) =  plot( red_vec/40e3,LaserStim,'Color',[ 0.4660 0.6740  0.1880]);
h(3:6) =  plot( [Lmove',Lmove']/40e3,[0,5],'k');
xlabel('Time [s]') ;     ylabel('Stimulus intensity [?]')
axis([0,par.N_last/4e4,1.5,3.5])
legend(h(1:3),'laser ON','laser stimulus','division line')
hold off
drawnow