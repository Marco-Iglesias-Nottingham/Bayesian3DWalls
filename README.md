# Bayesian3DWalls
Marco Iglesias, University of Nottingham, 2024. 
<br>

Here we provide data and files that were used used for the paper:
<br>
["Bayesian inversion for in-situ thermal characterisation of walls in the presence of thermal anomalies"](https://www.sciencedirect.com/science/article/pii/S0378778824006741)

The code is written to run on High Performance Computing platforms. Hence the `.sh` files in `Code` need to be modified accorgingly (e.g. add accounts, resources, etc). We used Tier 2 Midlands Sulis for our computations.

The code is written in python while the visualisation as well as the generation of the prior ensemble were done in MATLAB. 


## Reproducing the results:

Run the two MATLAB scripts provided in `MATLAB_Codes`

Run: `python main.sh 3D` 

Several output files will be stored in a folder name `Results` within `Code`. The folder needs to be moved to parent folder for visualisation. The main output file for visualistion is `Visual_3D.mat`. 

Ones this is completed you run the MATLAB file `GetFlowRates_for_Equiv.m` within the folder `FlowRates`. This will produce the posterior-informed flow rates from the 3D model which we need to build the equivalent 1D model from Section 5.

Run: `python main.sh 3D` 

