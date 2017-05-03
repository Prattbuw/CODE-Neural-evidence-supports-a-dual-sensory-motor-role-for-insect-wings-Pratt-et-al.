function [ M] = FRanalyzeMean( par,NE )
%FRanalyzeMean determines mean and projection of firing rate response to
%laser 
    % TMOHREN 2016-12-21
    % Last update 5/2/2017
    %------------------------------------


    for j = 1 :size(par.MothN,1)
        M.mean(:,j) = mean( NE.(['Nrate',num2str(j)]),2) / ...
            norm(mean( NE.(['Nrate',num2str(j)]),2));
        M.proj(j,:) = NE.(['Nrate',num2str(j)])'  * M.mean(:,j)/ ...
            norm( NE.(['Nrate',num2str(j)])'  * M.mean(:,j));
    end

end

