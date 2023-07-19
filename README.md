# Instantons-and-montecarlo-method-in-quantum-mechanics
The aim of this project is to implement montecarlo techniques in the study of a quantum mechanical problem. In this context we studied a non relativistic particle in a double well potential and tried to understand the role of tunneling events (instantons). The project is written in python with the implementation of numba JIT compiler in order to obtain a sensitive speed up of the execution time .

## Table of Contents 

 1. Requirements
 2. Structure
 3. Instructions
 4. Reference

## 1. Requirements

Please be sure to have installed the following python packages before proceeding further:
  1. numpy
  2. random
  3. tqdm
  4. os
  5. numba
  6. matplotlib.pyplot
     
## 2. Structure

The project is made up of several files:
 - `inputs.txt`: in this file there are the inputs for all the programs of the project. Parameters are already set with recommended values ;
 - `inputs.py`: module containing functions that read the input parameters from `inputs.txt` and pass them to the programs;
 - `function.py`: module containing all the relevant functions used in all the project's computations;
 -  `schroedinger.py`: module containing a schrodinger equation resolver using finite difference matrix method. Note that this is a general solver, it can be reused for other purposes;
 -   `qmdiag.py`: programs that exactly solve the schrodinger equation for the double well potential fining energy eigenvalues and eigenstates  and computes
correlation functions, log derivatives of corr. functions and the free energy. The spectrum for different values of the double well separation is computed in `spectrum.py`;
 -   `qm.py`: Montecarlo simulation of one non relativistic particle in a double well potential. This programme computes:  ground state wavefunction (squared),correlation functions and their log derivatives,  average quantities (action, position, energies....);
 -   `qmswitch.py`: this program computes the free energy of the anharmonic oscillator by means  of adiabatic switching. Reference system is harmonic oscillator
This program only evaluate the value of the free energy for a fixed  tmperature (euclidean time). Different values of free energy for different temperatures are computed in  `qmswitch_loop.py` which iterate the computations of `qmswitch.py`;

 - `qmcool.py`: Montecarlo implementation with cooling procedures. This programme computes the same things of qm.py but with cooled configuration. Analysis of instanton content is carried out. In particular in `qmcool_loop.py` are computed instanton density and action for instanton as function of cooling sweeps three differnt values of double well separation. In `qmcool_loop2.py`  is computed the instanton density as function of different well separations  after 10 cooling sweeps;

 - `qmidens.py`: computation of non gaussian corrections to the instanton density. This programme computes non gaussian corrections only for a fixed valeu of well separation. `qmidens_loop.py` iterates the procedure for different values;
 - `rilm.py` : this program computes correlation functions of the anharmonic oscillator using a random ensemble of instantons. \
        The multi-instanton configuration is constructed using the sum ansatz;
 - `rilmgauss.py`: this program generates the same random instanton ensemble as `rilm.py` but it also includes Gaussian fluctuations around the classical path. This is done by performing a few heating sweeps in the Gaussian effective potential;
 - `iilm.py`: this program computes correlation functions of the anharmonic oscillator using an interacting ensemble of instantons.  Very close instanton-anti-instanton pairs are excluded by adding  a nearest neighbor interaction with a repulsive core (hardcore interaction);
 - `streamline.py`: this program solve the streamline equation for an instanton-anti instanton pair using descent method;
 - `plots.py`: graphical analysis of data coming from the previous programs

The output datas of a given program are stored in .dat files in the directory 'Data/program_name'. The directory of a programme is created the first time the programme is runned. Also the plots are saved in such directories in .pdf and .png format.

# 3.Instructions

1. Install the required python packages
2. Download all files of this repository and place them in the same folder
3. Set input parameters in the file `inputs.txt` (recommended values are already set)
4. Run the programmes:
     -  `qmdiag.py`
     -  `qm.py`
     -   `qmswitch_loop.py` (for details of switching procedure run  `qmswitch.py`)
     -   `qmcool.py`
     -   `qmcool_loop.py`
     -   `qmcool_loop2.py`
     -   `qmidens.py`
     -   `qmidens_loop.py`
     -    `rilm.py`
     - `rilmgauss.py`
     -  `iilm.py`
     -   `streamline.py`
 5. For a complete graphical analysis run  `plots.py`

# 4.Reference
T. Schaefer, Instantons and Monte Carlo Methods in Quantum Mechanics, (2004) [hep-lat/0411010]
