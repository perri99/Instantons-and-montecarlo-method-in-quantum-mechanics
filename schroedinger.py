import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
import math
from tqdm import tqdm
import numba as nb

@nb.jit(nopython = True)
def anharmonic_potential(x):
    minimum, mass = initialize_potential()
    return 2*mass*(x**2-minimum**2)**2

@nb.jit(nopython = True)
def initialize_potential():
	minimum = 1.4
	mass = 0.5 # a.u.
	return minimum, mass

@nb.jit(nopython = True)
def initialize_param():
	x_min = -4
	x_max = +4  #integration range
	point_num = 1000   #lattice discretization
	return x_min, x_max, point_num

@nb.jit(nopython = True)
def normalization(x,dx):
    norm = np.sqrt(np.sum(x**2 * dx))
    return x/norm

@nb.jit(nopython = True)
def solve_schroedinger(mass: nb.float64, x: nb.float64[:], V: nb.float64[:], Step: nb.float64,point_num: nb.float64) -> nb.types.Tuple((nb.float64[:], nb.float64[:, :])):
    DiagConst = 2.0 / (2.0*mass*Step*Step)
    NondiagConst =  -1.0 / (2.0*mass*Step*Step)
    #Setting up tridiagonal matrix and find eigenvectors and eigenvalues
    Hamiltonian = np.zeros((point_num,point_num)) 
    Hamiltonian[0,0] = DiagConst + V[0] 
    Hamiltonian[0,1] = NondiagConst
    for i in range(1,point_num-1):
        Hamiltonian[i,i-1]  = NondiagConst
        Hamiltonian[i,i]    = DiagConst + V[i]
        Hamiltonian[i,i+1]  = NondiagConst
        
    Hamiltonian[point_num-1,point_num-2] = NondiagConst 
    Hamiltonian[point_num-1,point_num-1] = DiagConst + V[point_num-1]
    # diagonalize and obtain eigenvectors and eigenvalues, not necessarily sorted 
    EigValues, EigVectors = np.linalg.eig(Hamiltonian)

    # sort eigenvectors and eigenvalues
    permute = EigValues.argsort()
    EigValues = EigValues[permute]
    EigVectors = EigVectors[:,permute]
    #normalization of Eigenvectors
    for n in range(point_num):
        EigVectors[:,n] = normalization(EigVectors[:,n], Step)
    return EigValues, EigVectors

@nb.jit(nopython = True)
def correlation_function(E, e, x, dx, d, tau):
    G = 0
    for n in range(5):
        c = 0
        for j in range(len(x)):
            c += x[j]**d * E[j,0] * E[j, n] *dx
        G += c**2 * math.exp(-(e[n]-e[0]) * tau)
    return G

def log_derivative(x,n, dt):
    dx = np.zeros(n)
    for j in range(n):
        dx[j]  = (x[j]-x[j+1]) / (x[j]*dt) #log derivative
    return dx

if __name__ == '__main__':
    minimum, mass = initialize_potential()
    x_min, x_max, point_num = initialize_param()
    Step = (x_max-x_min)/(point_num - 1)
    x = np.linspace(x_min, x_max, point_num)  #creation of a lattice 1d space
    V = anharmonic_potential(x)
    EigValues, EigVectors = solve_schroedinger(mass, x, V, Step, point_num)
       