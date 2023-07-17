import numpy as np
import functions as fn
import random
from tqdm import tqdm
import inputs

'''
Montecarlo simulation of one non relativistic particle\
    in a double well potential.
This programme computes:
    1. ground state wavefunction (squared)
    2. correlation functions and their log derivatives
    3. average quantities (action, position, energies....)
'''

#-----inputs initialization-----------------------------------------|
f, n, a, neq, nmc, dx, n_p, nc, kp, mode, seed = inputs.qm()
tmax = n*a
random.seed(seed)
#-----output files--------------------------------------------------|
fn.directory('qm')

config1 = open('Data/qm/config1.dat', 'w')
config1.write('configuration, Action, Kinetic, Potential\n')

config2 = open('Data/qm/config2.dat', 'w')
config2.write('i*a, x[i]\n')

averages = open('Data/qm/averages.dat', 'w')
averages.write('Stot_av, S_tot_err, V_av, Verr,T_av, T_err, TV_av, TV_err\n')

correlations = open('Data/qm/correlations_qm.dat', 'w')
correlations.write("x correlation function\n")

correlations2 = open('Data/qm/correlations2_qm.dat', 'w')
correlations.write("x2 correlation function\n")

correlations3 = open('Data/qm/correlations3_qm.dat', 'w')
correlations3.write("x3 correlation function\n")

wavefunction = open('Data/qm/wavefunction_qm.dat', 'w')
wavefunction.write('x, psi(x)\n')
#-----histograms parameters-----------------------------------------|
nxhist = 50
xhist_min = -2.0*f
stxhist   = -2*xhist_min/nxhist
#-------counter variables-------------------------------------------|
nconf = 0 #it counts the number of randomly generated configurations 
ncor = 0 #it counts the number of correlations taken
#-------variable definitions----------------------------------------|
S_sum = 0.0
S2_sum = 0.0
T_sum = 0.0
T2_sum = 0.0
V_sum = 0.0
V2_sum = 0.0
TV_sum = 0.0
TV2_sum = 0.0
x_sum = 0
x2_sum = 0
x4_sum = 0
x8_sum = 0
#-----Array defnitions-----------------------------------------------|
xcor_sum   = np.zeros(n_p)         
xcor2_sum  = np.zeros(n_p) 
x2cor_sum  = np.zeros(n_p)
x2cor2_sum = np.zeros(n_p)
x3cor_sum  = np.zeros(n_p)
x3cor2_sum = np.zeros(n_p)
xcor_av    = np.zeros(n_p)
x2cor_av   = np.zeros(n_p)
x3cor_av   = np.zeros(n_p)
xcor_er    = np.zeros(n_p)
x2cor_er   = np.zeros(n_p)
x3cor_er   = np.zeros(n_p)
histo_x    = np.zeros(nxhist)
#starting configuration-----------------------------------------|
x = fn.periodic_starting_conf( n, f, mode)
#---------montecarlo generations--------------------------------|
for i in tqdm(range(nmc)):
    x = fn.periodic_update(x,n,a,f, dx)  #metropolis algorithm implementation with periodic boundary conditions
    if i < neq:
        continue
    nconf += 1
    #computation of action, kinetic energy, potential and virial for the current configuration
    S, V, T, TV = fn.compute_energy(x, n, a, f)
    S_sum += S
    S2_sum += S**2
    T_sum += T
    T2_sum += T**2
    V_sum += V
    V2_sum += V**2 
    TV_sum += TV
    TV2_sum += TV**2
    for ic in range(nc):
        ncor += 1 
        xcor = fn.correlations_functions(x, n, n_p)
        xcor_sum   = np.add(xcor, xcor_sum)
        xcor2_sum  = np.add(xcor**2, xcor2_sum)
        x2cor_sum  = np.add(xcor**2, x2cor_sum)
        x2cor2_sum = np.add(xcor**4, x2cor2_sum)
        x3cor_sum  = np.add(xcor**3, x3cor_sum)
        x3cor2_sum = np.add(xcor**6, x3cor2_sum)
        x_sum  += np.sum(x)
        x2_sum += np.sum(x**2)
        x4_sum += np.sum(x**4)
        x8_sum += np.sum(x**8)
    for k in range(n):
        fn.histogramarray(x[k], xhist_min, stxhist, nxhist, histo_x)
    #output configuration
    if i % kp == 0:
        config1.write("{}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(i, S, T, V))
        for j in range(n):
            config2.write("{:.4f}\t{:.4f}\n".format(j*a, x[j]))
        config2.write("------------------\n")
#----------end of montecarlo generations------------------------|
#----------Averages and error computations----------------------|
stot_av,stot_err = fn.dispersion(nconf,S_sum,S2_sum)
vtot_av,vtot_err = fn.dispersion(nconf,V_sum,V2_sum)
ttot_av,ttot_err = fn.dispersion(nconf,T_sum,T2_sum)
tvir_av,tvir_err = fn.dispersion(nconf,TV_sum,TV2_sum)
x_av,x_err       = fn.dispersion(nconf*n,x_sum,x2_sum)
x2_av,x2_err     = fn.dispersion(nconf*n,x2_sum,x4_sum)
x4_av,x4_err     = fn.dispersion(nconf*n,x4_sum,x8_sum)
for ip in range(n_p):
    xcor_av[ip],xcor_er[ip]   = fn.dispersion(ncor,xcor_sum[ip],xcor2_sum[ip])
    x2cor_av[ip],x2cor_er[ip] = fn.dispersion(ncor,x2cor_sum[ip],x2cor2_sum[ip])
    x3cor_av[ip],x3cor_er[ip] = fn.dispersion(ncor,x3cor_sum[ip],x3cor2_sum[ip])  
v_av   = vtot_av/tmax
v_err  = vtot_err/tmax
t_av   = ttot_av/tmax
t_err  = ttot_err/tmax
tv_av  = tvir_av/tmax
tv_err = tvir_err/tmax
e_av   = v_av + tv_av
e_err  = np.sqrt(v_err**2 + tv_err**2)
#output averages
averages.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(stot_av,stot_err, v_av, v_err, t_av,tv_av,tv_err)) 
averages.write('e_av, e_err, x_av, x_err, x2_av,x2_err, x4_av, x4_err\n')  
averages.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(e_av, e_err, x_av, x_err, x2_av, x2_err, x4_av, x4_err))
#----------log derivatives------------------------------------------------------------|
dx, dxe = fn.log_derivatives(xcor_av, xcor_er, a)
x2sub_av, x2sub_er = fn.substract(x2cor_av, x2cor_er) #substracting: in the correlators there are constant terms to be substracted in the log computations
dx2, dxe2 = fn.log_derivatives(x2sub_av, x2sub_er, a)
dx3, dxe3 = fn.log_derivatives(x3cor_av, x3cor_er, a)
#output log derivatives
for ip in range(n_p-1):
    correlations.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, xcor_av[ip], xcor_er[ip], dx[ip], dxe[ip]))
    correlations2.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x2cor_av[ip], x2cor_er[ip], dx2[ip], dxe2[ip]))
    correlations3.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format(ip*a, x3cor_av[ip], x3cor_er[ip], dx3[ip], dxe3[ip]))
#----------wave function---------------------------------------------------------------|
position, wave_function = fn.building_wavefunction(histo_x, nxhist, stxhist, xhist_min)
for i in range(nxhist):
    wavefunction.write("{:.4f}\t{:.4f}\n".format(position[i], wave_function[i]))
#-----------closing files--------------------------------------------------------------|
config1.close()
config2.close()
averages.close()
correlations.close()
correlations2.close()
correlations3.close()
wavefunction.close()