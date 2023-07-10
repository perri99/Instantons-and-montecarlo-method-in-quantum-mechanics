import numpy as np
import matplotlib.pyplot as plt
import schroedinger as sc
from tqdm import tqdm

def double_well(x,f):
    return (x**2-f**2)**2

#-----------spectrum with different f------------------------------------------|
minimum, mass = sc.initialize_potential()
x_min, x_max, point_num = sc.initialize_param()
x = np.linspace(x_min, x_max, point_num)
Step = (x_max-x_min)/(point_num - 1) 
f = np.linspace(0,2,10)
eigenvals = np.zeros((6, 10))
for j in tqdm(range(10)):
    V = double_well(x, f[j])
    e, E = sc.solve_schroedinger(mass, x, V, Step, point_num)
    for i in range(6):
        eigenvals[i,j] = e[i]
for i in range (6):
    plt.plot(f, eigenvals[i,:])
plt.show()
