# Pratt_LaserStim
Laser experiment analysis for Pratt et al. 
% ------------------
TL Mohren 
Last update: 5/2/2017
%--------------------

To run analysis of laser experiment
	1) got to folder /scripts 
	2) open Run_laseranalysis
	3) 	Insect location of data as par.datafolder = 'C:\bla...\...bla\data'
		Determine which moth to analyze for in par.w_moths (could be all =1:33, or specify moth =17), currently set to all. 
		Determine if you want diagnostic figures (intermediate steps to check) 
		for yes, par.diagnostic_fig = 1;
		for no, par.diagnostic_fig = 0;
	4) run code (should take in the order of 100 - 1000 seconds) 
