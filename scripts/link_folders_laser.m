%------------------------------------
% Links folders
% TMOHREN 2016-12-21
% Last update 5/2/2017
%------------------------------------
script_dir = fileparts(mfilename('fullpath'));
base_dir = script_dir(1:(end-8));
cd(base_dir);

addpath([base_dir '\functions'])
addpath(par.datafolder)
addpath([base_dir '\scripts'])
