function [ NE ,F] = LaserFiringRate( par,Sp,L )
%LaserFiringRate extracts firing rate from spikes during laserphase
    % TMOHREN 2016-12-21
    % Last update 5/2/2017
    %------------------------------------

fprintf('Calculating firing rate...  \n');
tic 
F.drate = zeros( size(par.MothN,1),par.N_last );
if par.diagnostic_fig
    f = figure(2);
        set(f,'name','Discrete firing rate ')
        f.Position = [100,100,800 900];
end

for j = 1:size(par.MothN,1)
    SpikeInd = round(  nonzeros(    Sp.(['M',num2str(par.MothN(j,1))])(:,par.MothN(j,2))      )*4e4  );
    SpikRel = SpikeInd(SpikeInd<par.N_last );
    if numel(SpikRel) ~=0
        clear dt
        dt(2:length(SpikRel)) = SpikRel(2:end) - SpikRel(1:end-1);
        dt(1) = SpikRel(1)+par.N_last -SpikRel(end);
        for j2 = 2:length(dt);
           F.drate(j,SpikRel(j2-1):SpikRel(j2)) = 4e4/dt(j2); 
        end
        F.drate(j,1:SpikRel(1)-1) =  4e4/dt(1);
        F.drate(j,SpikRel(end)+1:par.N_last ) =  4e4/dt(1);
    else
        F.drate(j,1:par.N_last ) = 0;
    end
    if par.diagnostic_fig
        subplot(8,8,j);
        plot( F.drate(j,1:1000:length(F.drate)) )
        drawnow
    end
    fprintf('     Moth %d, neuron %d : firing rate done \n',par.MothN(j,1),par.MothN(j,2));
end
display('Discrete firing rate done')

%  Smooth firing rate with Gaussian 
% contains Gaussian Gs .r(egular)   .c(alibrated)   .s(hifted)  
Pred    = 1e4;                          % undersample firing rate to increase speed 
Pvec    = 1:Pred:par.N_last ;           % undersampled indices 
F.crate = zeros(size(par.MothN,1),length(Pvec));
Gsw     = 40e3;                       % width of gaussian convolution 
F.local = zeros(size(par.MothN,1),26,60); 
Gss = zeros(1,length(Pvec));            
 
for j = 1:size(par.MothN,1)
    Nj = length(Pvec);
    for k = 1:Nj 
        Im = floor(Nj/2);
        Gsr = exp(-(Pvec-Pvec(Im)).^2./Gsw^2);
        Gsc = Gsr /sum(Gsr);
        if k<Im
            Gss(1:k+Im) = Gsc(Nj-k-Im+1:Nj);
            Gss(k+Im+1:Nj) = Gsc(1:Nj-k-Im);
        elseif k ==Im
            Gss = Gsc;
        elseif k>Im
            Gss(1:Im-(Nj-k)) =  Gsc(Nj-k+Im+1:Nj); 
            Gss(Im-(Nj-k)+1:Nj) =  Gsc(1:Nj-k+Im); 
        end
        F.crate(j,k) = sum(F.drate(j,Pvec).*Gss);
        if mod(k,200) ==0
            stimn = par.MothN(j);
            if par.diagnostic_fig   % plot drawing the gaussian 
                run('diagnostic_LaserFiringRateConvolution')
            end
        end        
    end
    
    stimn = par.MothN(j);
    for g = 1:L.pulses(stimn)
       stimrange = round(  L.st(stimn,g)-8e4 :Pred: L.end(stimn,g) + 32e4,-4);
       stimr = length( nonzeros(round((stimrange/Pred)) ));
       F.local(j,g,1:stimr  ) = F.crate(j,nonzeros(round((stimrange/Pred)) ) );
    end 
    
    Fcal = (  squeeze(F.local(j,:,1:60))  -...      % remove mean for Fcalibrated 
        repmat(  mean(squeeze(F.local(j,:,1:60)) ,2),1, 60))';
    NE.(['Nrate',num2str(j)]) = Fcal;
    
    if par.diagnostic_fig       
      run('diagnostic_LaserFiringRateEnsemble')
    end
end

for j = 1:size(par.MothN,1)
    NE.(['N',num2str(j)]).Sp = nonzeros(Sp.(['M',num2str(par.MothN(j,1))])(:,par.MothN(j,2)));
end
fprintf('Firing rate analyzed, took %d seconds \n',round(toc,0))

end

