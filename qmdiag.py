import numpy as np
import schroedinger as sc
from functions import directory
from tqdm import tqdm
import matplotlib.pyplot as plt
 
#------------------------output files------------------------------------|
directory('qmdiag')
write_groundstate = open('Data/qmdiag/ground_state1_doublewell.dat', 'w')
write_eigenvalues = open('Data/qmdiag/eigenvalues_doublewell.dat', 'w')
cor = open('Data/qmdiag/cor.dat', 'w')
cor2 = open('Data/qmdiag/cor2.dat', 'w')
cor3 = open('Data/qmdiag/cor3.dat', 'w')
write_dlog = open('Data/qmdiag/dlog.dat', 'w')
partition_function = open('Data/qmdiag/partition_function.dat', 'w')

tmax = 2.5
ntau = 100
dtau = tmax/float(ntau)

xcor = np.zeros(ntau+1)
x2cor = np.zeros(ntau+1)
x3cor = np.zeros(ntau+1)

minimum, mass = sc.initialize_potential()
x_min, x_max, point_num = sc.initialize_param()
Step = (x_max-x_min)/(point_num - 1)

x = np.linspace(x_min, x_max, point_num)  #creation of a lattice 1d space
V = sc.anharmonic_potential(x)
EigValues, EigVectors = sc.solve_schroedinger(mass, x, V, Step, point_num)
ground_state = EigVectors[:,0]

#------------correlations  functions-------------------------------------------|
for i in tqdm(range(ntau+1)):
    tau = i * dtau
    xcor[i] = sc.correlation_function(EigVectors, EigValues, x, Step, 1, tau)
    cor.write("{:.4f}\t{:.4f}\n".format(tau, xcor[i]))
    
    x2cor[i] = sc.correlation_function(EigVectors, EigValues, x, Step, 2, tau)
    cor2.write("{:.4f}\t{:.4f}\n".format(tau, x2cor[i]))
    
    x3cor[i] = sc.correlation_function(EigVectors, EigValues, x, Step, 3, tau)
    cor3.write("{:.4f}\t{:.4f}\n".format(tau, x3cor[i]))

#------------log derivative of <x^n(0)x^n(t)>--------------------------------------|
add = np.full(ntau+1, -x2cor[ntau])
dlog = sc.log_derivative(xcor, ntau, dtau)
dlog2 = sc.log_derivative(np.add(x2cor,add), ntau, dtau)
dlog3 = sc.log_derivative(x3cor, ntau, dtau)
for i in range(ntau):
    tau = i * dtau
    write_dlog.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(tau, dlog[i], dlog2[i], dlog3[i]))
    
#------------partition function------------------------------------------------|
xlmax = 100.0
xlmin = 0.1
xlogmax = np.log(xlmax)
xlogmin = np.log(xlmin)
nl = 50
dlog = (xlogmax-xlogmin)/float(nl)
partition_function.write("T\t beta\t F\n")
for il in range(nl+1):
    xlog = xlogmin+il*dlog
    xl = np.exp(xlog) #questo è il tempo euclideo
    t  = 1.0/xl #quetsa è la temperatura(inverso del tempo euclideo)
    z  = 1.0
    for i in range(1, point_num):
        z += np.exp(-(EigValues[i]-EigValues[0])*xl) #partition function
    p = t*np.log(z) - EigValues[0]                   #free energy
    partition_function.write("{:.4f}\t{:.4f}\t{:.4f}\n".format(t, xl, p))
#------------writing wavefunction of ground state and eigenvalues--------------|
write_eigenvalues.write('State |n> '+' Eigenvalue\n')
for j in range(10):
    write_eigenvalues.write("{}\t{:.4f}\n".format(j, EigValues[j]))

write_groundstate.write('x '+'psi_0(x)\n')
for j in range(point_num):
    write_groundstate.write("{:.4f}\t{:.4f}\n".format(x[j], ground_state[j]))

plt.plot(x, ground_state, label = 'groundstate')
plt.xlabel('x')
plt.ylabel('Psi0(x)')
plt.legend()
plt.savefig('Data/qmdiag/groundstate.pdf')
plt.savefig('Data/qmdiag/groundstate.png')
plt.show()
#------------plotting----------------------------------------------------------|
plt.plot(x,V)
for i in range(4):
    plt.plot(x, EigVectors[:,i]**2 + EigValues[i], label = 'state '+str(i))
plt.title('First 3 eigenvectors squared')
plt.xlabel('x')
plt.ylabel('Psi(x)')
plt.ylim(0,10)
plt.legend()
plt.savefig('Data/qmdiag/eigenvectors.pdf')
plt.savefig('Data/qmdiag/eigenvectors.png')
plt.show()
#-----------closing file-------------------------------------------------------|
partition_function.close()
write_eigenvalues.close()
write_groundstate.close()
cor.close()
cor2.close()
cor3.close()
write_dlog.close()