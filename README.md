# NAPL
A neuro-adaptive programming language for general-purpose neuromorphic computing

## Installation
All provided installation methods allow running ```napl``` in the command line and ```import napl``` as a python module.

Make sure you have [Anaconda](https://www.anaconda.com/) installed before the steps below.

### Option 1: pip installation
1. ```git clone``` [this repo](https://github.com/UnaryLab/napl) and ```cd``` to the repo dir.
2. ```conda env create -f environment.yaml```
   - The ```name: napl``` in ```evironment.yaml``` can be updated to a preferred one.
3. ```conda activate napl```
4. ```pip install napl```
5. Validate installation via ```napl -h``` in the command line or ```import napl``` in python code

### Option 2: source installation
This is the developer mode, where you can edit the source code with live changes reflected for simulation.
1. ```git clone``` [this repo](https://github.com/UnaryLab/napl) and ```cd``` to the repo dir.
2. ```conda env create -f environment.yaml```
3. ```conda activate napl```
4. ```python3 -m pip install -e . --no-deps```
5. Validate installation via ```napl -h``` in the command line or ```import napl``` in python code
