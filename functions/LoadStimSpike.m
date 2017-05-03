function [ Sp,par ] = LoadStimSpike( par)
%LoadStimSpike loads data into a struct 
    % TMOHREN 2016-12-21
    % Last update 5/2/2017
    %------------------------------------

fprintf('Loading data...  \n');

tic 
for j = par.w_moths
    if exist( [par.SpikeName{j} ,'.txt'] )            % load spike times, 
        Sp.(['M',num2str(j)]) = csvread( [par.SpikeName{j} ,'.txt']  , 1,0);
        
        % find if columns are doubled 
        if Sp.(['M',num2str(j)])(:,1) == Sp.(['M',num2str(j)])(:,2)
            display('this works' )
        end
        
        % checks for last column containing i 
        fid = fopen([par.SpikeName{j} ,'.txt'] );
        C = textscan(fid ,'%s');
        fclose(fid);
        firstline =char(C{1,1}(1));
        if firstline(end) == 'i'
            display(['Discard last column of moth ' ,num2str(j) ])
            Sp.(['M',num2str(j)])(:,end) = [];   % clears last column
        end
    else
        Sp.SpikeName{j} = 'Not Found';
        display([ par.SpikeName{j} ,'.txt, does not exist' ]);
    end
    
    if exist( [par.StimName{j} '.txt' ])             %  load Stimulus 
        fileID = fopen( [ par.StimName{j} '.txt']);
        formatSpec = '%f %f';
        TempSt = textscan(fileID,formatSpec,par.N_last,'Delimiter','\t'); % load stimulus up to par.N_last (end of laserphase)
        fclose(fileID);
        Sp.(['M',num2str(j),'stim']) = TempSt{2};                         % If file contains 2 columns, 2nd column is stimulus (1st is raw recording) 
    elseif exist( [par.StimName{j} '.mat' ])
        Temp = load(par.StimName{j});
        Sp.(['M',num2str(j),'stim']) = Temp.Stim(1:par.N_last);
    else
        display([par.StimName{j} ' not found' ])
    end
end

par.MothN = [];
for j = par.w_moths
    Neurs = size(Sp.(['M',num2str(j)]),2);    % subract 1 neuron 
    par.MothN = [par.MothN;  [repmat(j,Neurs,1), (1:Neurs)'] ];
end
fprintf('Data loaded, loading took %d seconds \n',round(toc,0))

end

