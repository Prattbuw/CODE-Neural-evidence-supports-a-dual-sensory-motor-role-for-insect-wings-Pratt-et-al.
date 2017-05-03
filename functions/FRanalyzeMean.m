function [ M] = FRanalyzeMean( par,NE )
%UNTITLED8 Summary of this function goes here
%   Detailed explanation goes here

    % ADJUST, subtract individual mens 
    
    for j = 1 :size(par.MothN,1)
        M.mean(:,j) = mean( NE.(['Nrate',num2str(j)]),2) / ...
            norm(mean( NE.(['Nrate',num2str(j)]),2));
        M.proj(j,:) = NE.(['Nrate',num2str(j)])'  * M.mean(:,j)/ ...
            norm( NE.(['Nrate',num2str(j)])'  * M.mean(:,j));
    end

end

