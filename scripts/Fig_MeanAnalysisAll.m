


Lwidth = 1e5;
Loff = Lwidth + L.end(par.MothN(j,1),1)-L.st(par.MothN(j,1),1);
Lplotind = 1 : Pred : (Loff+3e5)   ;
Lbinon = (  logical((Lplotind >= Lwidth) .* (Lplotind <=Loff) )   );
Lindoff  = find(Lbinon,1,'last');
Lindon = find(Lbinon,1); 
% Lcol = ['b';'c';'r';'g';'m'];
Lcol = linspecer(5);
location_names = {'base','2','3','4','control'};
%     
for j = 1:size(par.MothN,1)
    if (j-1)/10 ==  floor((j-1)/10)
        fsheet = figure('Position', [50,50,1200 950] );
        subplot(10,4,1)
        title(['Unsorted neuron sheet using mean ',num2str(floor((j-1)/10)+1)])
        hold on 
    end
    subplot(10,4,( j-1- (floor((j-1)/10))*10)*4+1    )
        fill( [Lindon,Lindon,Lindoff,Lindoff] ,...
                [0,1,1,0]-0.5,...
                [0.8,0.8,0.8],'EdgeColor','none')    
        hold on 
        temp = NE.(['Nrate',num2str(j)])(:,:)./norm(NE.(['Nrate',num2str(j)])(:,:));
%         plot(temp(:,1:5)*3,'--b','LineWidth',[1])
        plot(M.mean(:,j),'LineWidth',[2.5])
        legend('Laser on','Mean firing rate','Location','NorthEastOutside')
        axis([0,60,-0.5,0.5])
    subplot(10,4,( j-1- (floor((j-1)/10))*10)*4+2  )
        hold on 
        for k = 1:5
            bar( find( L.cat( par.MothN( j   ,1),:) ==k)    ,...
            M.proj(j,find( L.cat( par.MothN(j,1),:) ==k )   )  ,...
            'FaceColor',Lcol(k,:),'EdgeColor','none')
        end
        axis([0,26,-0.5,0.5])
    subplot(10,4,( j-1- (floor((j-1)/10))*10)*4+3  )
        plot( (1:2.05e4)/2e4, F.drate(j,1:1000:length(F.drate)) )
    subplot(10,4,( j-1- (floor((j-1)/10))*10)*4+4  )        
        text(0.2,0.9,['n of spikes Mechanical phase = ',num2str( sum(NE.(['N',num2str(j)]).Sp*4e4>par.N_last))]);
        text(0.2,0.5,['n of spikes Laserphase = ',num2str( sum(NE.(['N',num2str(j)]).Sp*4e4<par.N_last))]);
        text(0.2,0.1,['M',num2str(par.MothN(j,1)),' N',num2str(par.MothN(j,2))]);
        axis off
    if ((j)/10 ==  floor((j)/10)) || j==size(par.MothN,1)
        saveas(fsheet,['figs/',par.figname,'_sheet',num2str(ceil((j)/10))],'png')
    end
end
