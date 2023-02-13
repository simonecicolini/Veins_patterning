# Veins_patterning
This repository contains source codes to simulate the signalling network involved in veins refinement and described in the paper "Signalling-dependent refinement of cell fate choice during tissue remodelling".
***
**Simulation on the wing tissue template**

The script ```Wing_simulation.py``` perform a simulation on the experimental wing. To run ```Wing_simulation.py``` one has to import ```tissue_miner.py``` and ```tissue_miner_tools.py```, which contain classes and functions to extract data from segmented wing movies. Data from wing movies can be found in the folder "wing_movies" at https://zenodo.org/record/7625645#.Y-oULi8w3gh, which contains both wild-type and Dumpy mutant wings. There are lines of code that can be commented/uncommented to simulate wild-type/dumpy mutant/optogenetic veins. To run the simulation it is sufficient to save the folder "wing_movies", ```tissue_miner_tools.py```, and ```tissue_miner.py``` in the same directory as ```Wing_simulation.py```.
***
**Simulation on hexagonal cells**

The script ```Hexagon_simulation.py``` perform a simulation of vein refinement on a hexagonal lattice. It does not require to import other files and can be readily run. 
