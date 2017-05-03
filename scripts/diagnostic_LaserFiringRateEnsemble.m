%------------------------------------
% diagnostic fig, can be toggled off in main script 
% TMOHREN 2016-12-21
% Last update 5/2/2017
%------------------------------------
f = figure(4);
            set(f,'name','Firing rate during laser')
            f.Position = [1200,100,500 900];
        clf(4)
        for g = 1:L.pulses(stimn)
           stimrange = round(  L.st(stimn,g)-5e4 :Pred: L.end(stimn,g) + 3e5,-4);
           r = L.cat(stimn,g);
           c = sum( L.cat(stimn,1:g) == L.cat(stimn,g) );
           subplot(5,6,c + 6*(r-1))
               h(1) = fill([L.st(stimn,g);L.st(stimn,g);L.end(stimn,g);L.end(stimn,g)],...
                         [0,1,1,0]*200,...
                         [0.8,0.8,0.8],'EdgeColor','none');
               hold on
               h(2) = plot( stimrange( find(round(stimrange/Pred))  ), ...
                   F.crate(j,nonzeros(round((stimrange/Pred)) ) ) );
               uplim = max(max(F.local(j,:,:)));
               if uplim > 1
                    axis([stimrange(1),stimrange(end),0,uplim])
               else
                    axis([stimrange(1),stimrange(end),0,1])
               end
                    set(gca,'XTickLabel',{})
                    hold off
        end
        subplot(5,6,1);ylabel('Wing base')% ,'rot',0
            title(['# ',num2str(j)])
        subplot(5,6,7);ylabel('Near wing base')% ,'rot',0
        subplot(5,6,13);ylabel('Wing vein')% ,'rot',0
        subplot(5,6,19);ylabel('Wing tip')% ,'rot',0
        subplot(5,6,25);ylabel('Zero case')% ,'rot',0
        legend(h(1:2),'Laser ON','Firing rate')
        drawnow