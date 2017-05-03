%------------------------------------
% Diagnostic figure
% TMOHREN 2016-12-21
% Last update 5/2/2017
%------------------------------------
j = 1;
Pred = 1e4;
Pvec = 1:Pred:par.N_last ;
Lcol = linspecer(5);

% if 
stimn = par.MothN(j);

if par.diagnostic_fig
    figure('Position', [50,50,1300 950])
    subplot(311)
    for q = 1:L.pulses(stimn)
         h(1:6) = fill([L.st(stimn,q);L.st(stimn,q);L.end(stimn,q);L.end(stimn,q)]/4e4,...
             [0,1,1,0]*200,...
             [0.8,0.8,0.8],'EdgeColor','none');
            hold on
    end
    plot( Pvec/4e4 , F.crate(j,:),'k','LineWidth',[1])
     for g = 1:L.pulses(stimn)
         stimrange = round(  L.st(stimn,g)-5e4 :Pred: L.end(stimn,g) + 3e5,-4);
          r = L.cat(stimn,g);
          c = sum( L.cat(stimn,1:g) == L.cat(stimn,g) );
              tempt = stimrange( find(round(stimrange/Pred))  )/4e4;
              tempf = F.crate(j,nonzeros(round((stimrange/Pred)) ) );
%               plot( tempt(1:35), ...
%                    tempf(1:35),'Color',Lcol(L.cat(par.MothN(j,1),g),:),'LineWidth',[3]);
              plot( tempt(1:45), ...
                   tempf(1:45),'Color',Lcol(L.cat(par.MothN(j,1),g),:),'LineWidth',[3]);
%               plot( tempt, ...
%                    tempf,'Color',Lcol(L.cat(par.MothN(j,1),g),:),'LineWidth',[3]);
     end
    xlabel('Time [s]')
    ylabel('Firing rate [f/s]')
    axis([0,500,0,max(F.crate(j,:))*1.2])
    subplot(335)
       h(1) = fill( ([5;5;25;25]-5)/4,...
                 [0,1,1,0]*40-20,...
                 [0.8,0.8,0.8],'EdgeColor','none');
         hold on
        stimrange = round(  L.st(stimn,g)-5e4 :Pred: L.end(stimn,g) + 3e5,-4);
        t_temp = (-10:1:49)/4;
        plot(t_temp,M.mean(:,j),'k','LineWidth',4)
        axis([-10/4,50/4,min(M.mean(:,j)*1.1),max(M.mean(:,j)*1.1)])
        xlabel('Time [s]')
        ylabel('Frate [f/s]' )

    subplot(313)
      for g = 1:L.pulses(stimn)
%           stimrange = round(  L.st(stimn,g)-5e4 :Pred: L.end(stimn,g) + 3e5,-4);
%            r = L.cat(stimn,g);
%            c = sum( L.cat(stimn,1:g) == L.cat(stimn,g) );
%            h(1) = fill([L.st(stimn,g);L.st(stimn,g);L.end(stimn,g);L.end(stimn,g)]/4e4,...
%                      [0,1,1,0]*M.proj(j,g)/norm(M.proj),...
%                      Lcol(L.cat(par.MothN(j,1),g),:),'EdgeColor','none');
           hold on
        for k = 1:5
%             bar( find( L.cat( par.MothN( j   ,1),:) ==k)    ,...
            bar( L.st(par.MothN(j,1), find( L.cat( par.MothN( j   ,1),:) ==k),:)/4e4    ,...
            M.proj(j,find( L.cat( par.MothN(j,1),:) ==k )   )  ,...
            'FaceColor',Lcol(k,:),'EdgeColor','none')
        end
      end
        xlabel('Time [s]' )
        ylabel(' $\frac{X \cdot \bar{X}}{|X \cdot \bar{X}|}$','Interpreter','LaTex' )
      plot([0,500],[0,0],'k','LineWidth',[0.7])
      axis([0,500,-0.5,0.5])
end