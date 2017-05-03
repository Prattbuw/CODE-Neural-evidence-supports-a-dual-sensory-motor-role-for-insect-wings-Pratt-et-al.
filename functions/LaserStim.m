function [ L] = LaserStim( par,Sp )
%LaserStim determines when the laser is on
    % TMOHREN 2016-12-21
    % Last update 5/2/2017
    %------------------------------------


fprintf('Analyzing laser stimulus...  ');
tic 

Lred = 1e2;                             % resample at 1/100
red_vec = 1:Lred:par.N_last;            % resample vector to increase speec 
Ldur = round(((5000-30)*40)/Lred);      % duration of laserpulse

% Lmove = (1:4)*0.4*1e7;     
Lmove = (100:100:400)*4e4;              %  moments when laser switched position 

% initialize start, stop, and category matrices 
L.st = zeros(length(par.w_moths),25);
L.end = zeros(length(par.w_moths),25);
L.cat = zeros(length(par.w_moths),25);
        
% some very particular conditions to find start and end of laser 
for j = par.w_moths
    Count = 1;
    LaserStim = Sp.(['M',num2str(j),'stim'])(red_vec);
    LaserSwitch = LaserStim;
    LaserSwitch( LaserSwitch <2.5) = 0;  
    LaserSwitch( LaserSwitch >2.5) = 1;
    for k = 501:length(LaserSwitch)-Ldur-3e3
       if any( LaserSwitch(  k-   Ldur/4  +1   :k-1)) == 0 && LaserSwitch(k) > 0.99
           L.st(j,Count) = red_vec(k);
           LaserSwitch(k+1:k+Ldur) = 1;
       elseif   any( LaserSwitch(k+1 :  k+   Ldur/4   )) == 0 && LaserSwitch(k) > 0.99
           L.end(j,Count) = red_vec(k) ;
           if L.end(j,Count)-L.st(j,Count) <= 2e5
                Count = Count + 1;
           end
       end
    end
    L.pulses(j) = Count-1;
    if par.diagnostic_fig 
        run('diagnostic_LaserStim' )
    end
end


% Split up laser in categories (L.cat), for the 5 different laser locations
L.cat = L.end<=Lmove(1);
L.cat = L.cat + ( ( L.end>=Lmove(1)) .*  (L.end<=Lmove(2) )  ) * 2;
L.cat = L.cat + ( ( L.end>=Lmove(2)) .*  (L.end<=Lmove(3) )  ) * 3;
L.cat = L.cat + ( ( L.end>=Lmove(3)) .*  (L.end<=Lmove(4) )  ) * 4;
L.cat = L.cat + ( ( L.end>=Lmove(4))   ) * 5;
L.cat( find(L.end== 0) ) = 0;

% display('Laser start + stop done')
fprintf('Laser stimulus analyzed, took %d seconds \n',round(toc,0))


end

