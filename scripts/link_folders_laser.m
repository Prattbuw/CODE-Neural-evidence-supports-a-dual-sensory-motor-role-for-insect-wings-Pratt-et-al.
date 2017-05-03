% link folders 

script_dir = fileparts(mfilename('fullpath'));
base_dir = script_dir(1:(end-8));
cd(base_dir);

addpath([base_dir '\functions'])
addpath(par.datafolder)
addpath([base_dir '\scripts'])


% Dat_loc = 'C:\Users\Daniellab\Desktop\Revised Stim\';