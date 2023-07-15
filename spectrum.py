import numpy as np
import matplotlib.pyplot as plt
import schroedinger as sc
from functions import directory
from tqdm import tqdm
'''
This programme computes the spectrum of the double well potential\
    in function of different values of double well separation f

'''

#--------------output file------------------------------------------------------
directory('qmdiag')
split = open('Data/qmdiag/splitting.dat', 'w')
#------------------------------------------------------------------------------|
#-----------spectrum with different f------------------------------------------|
x_min, x_max, minimum, mass, point_num = sc.initialize_param()
Step = (x_max-x_min)/(point_num - 1)

x = np.linspace(x_min, x_max, point_num)
f = np.linspace(0,3,50)

splitting = np.zeros(50)
eigenvals = np.zeros((6, 50))
for j in tqdm(range(50)):
    V = sc.anharmonic_potential(x, mass, f[j])
    e, E = sc.solve_schroedinger(mass, x, V, Step, point_num)
    splitting[j] = e[1] - e[0]
    for i in range(6):
        eigenvals[i,j] = e[i]
for i in range (6):
    plt.plot(f, eigenvals[i,:], label = 'Eigenvalue '+str(i))

plt.title('First six eigenvalues')    
plt.xlabel('f: well separation')
plt.ylabel('Energy')
plt.savefig('Data/qmdiag/spectrum.pdf')
plt.savefig('Data/qmdiag/spectrum.png')
plt.show()
plt.plot(f, splitting/2)
plt.xlabel('f: well separation')
plt.ylabel('Delta E/2')
plt.savefig('Data/qmdiag/splitting.pdf')
plt.savefig('Data/qmdiag/splitting.png')
plt.show()
split.write('f\t E1-E0\n')
for k in range(50):
    split.write('{:.4f}\t{:.4f}\n'.format(f[k], splitting[k]))
split.close()