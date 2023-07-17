import numpy as np
import functions as fn
import random
from tqdm import tqdm
'''
Computation of the free energy of the anharmonic oscillator by means\
    of adiabatic switching: S = S0 + alpha(S-S0)
    Reference system is harmonic oscillator
This program only evaluate the value of the free energy for a fixed\
    temperature (euclidean time). For a complete evaluation see\
        qmswitch_loop.py
Details of adiabatic switching procedure are saved in the output file
'''
def setting_inputs():
    f = 1.4 #minimum of the potential
    n = 800 #lattice points
    a = 0.05 #lattice spacing
    neq = 100 #number of equilibration sweeps
    nmc = 10**4 #number of MonteCarlo sweeps
    dx = 0.5 #width of updates
    n_alpha = 20 #number of switch
    nc = 5 #number of correlator measurements in a configuration
    kp = 50 #number of sweeps between writeout of complete configuration 
    mode = 0 # ih=0: cold start, x_i=-f; ih=1: hot start, x_i=random
    seed = 597
    w0 = 5.6
    return f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0

#------------------------setting inputs-----------------------------------
f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0 = setting_inputs()
random.seed(seed)
#------------------------output files------------------------------------
fn.directory('qmswitch')
        
switch = open('Data/qmswitch/switch.dat','w')
switch.write('montecarlo switch\n ----- \n')
switch.write('f\t n\t a\t nmc\t neq\t dx\t mode\t w0\t n_alpha\n')
switch.write("{:.2f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n\n".format(f,n,a, nmc, neq, dx, mode, w0, n_alpha))
#--------------------------counter----------------------------------
nconf = 0
#------------------------definitions--------------------------------
dalpha = 1.0/float(n_alpha)
beta   = n*a
e0     = w0/2.0   #energy of ground state of harmonic oscillator
f0     = 1.0/beta*np.log(2.0*np.sinh(e0*beta)) #harmonic oscillator free energy
ei     = e0      
#-------------------variable definitions----------------------------------------|
S_sum = 0.0
S2_sum = 0.0
T_sum = 0.0
T2_sum = 0.0
V_sum = 0.0
V2_sum = 0.0
Valpha_sum = 0.0
Valpha2_sum = 0.0
x_sum = 0
x2_sum = 0
x4_sum = 0
x8_sum = 0
#-----Array defnitions-----------------------------------------------|
Va_av = np.zeros(2*n_alpha)
Va_err = np.zeros(2*n_alpha)
#starting configuration-----------------------------------------|
x = fn.periodic_starting_conf(n, f, mode)
S, V, P = fn.compute_energy_switch(x, n, a, f, w0, 0)
#--------loop over coupling constant alpha-------------------------------------|
for ialpha in tqdm(range(2 * n_alpha)):
    if ialpha <= n_alpha:
        alpha = ialpha * dalpha  #da 0 a 1
    else:
        alpha = 2.0 - ialpha * dalpha #da 1 a 0
    #---------montecarlo generations-------------------------------------------|
    for i in range(nmc):
        nconf += 1
        if i == neq:  
            nconf = 0#when we reach equilibrium all is set to zero
            S_sum = 0.0
            S2_sum = 0.0
            T_sum = 0.0
            T2_sum = 0.0
            V_sum = 0.0
            V2_sum = 0.0
            Valpha_sum = 0.0
            Valpha2_sum = 0.0
            x_sum = 0
            x2_sum = 0
            x4_sum = 0
            x8_sum = 0        
        x = fn.update_periodic_switch(x,n,a,f, w0, alpha, dx)
        #computation of action, kinetic energy, potential and virial for the current configuration
        S, V, P = fn.compute_energy_switch(x, n, a, f, w0, alpha)
        S_sum  += S
        S2_sum += S**2
        V_sum  += V/beta
        V2_sum += V**2/beta
        Valpha_sum  += P/beta
        Valpha2_sum += P**2/beta
        for k in range(n):
            x_sum  += x[k]
            x2_sum += x[k]**2
            x4_sum += x[k]**4
            x8_sum += x[k]**8
    #----------end of montecarlo cicle-----------------------------------------|
    stot_av, stot_err     = fn.dispersion(nconf, S_sum, S2_sum)
    v_av, v_err           = fn.dispersion(nconf, V_sum, V2_sum)
    valpha_av, valpha_err = fn.dispersion(nconf,Valpha_sum, Valpha2_sum)
    x_av, x_err           = fn.dispersion(nconf*n, x_sum, x2_sum)
    x2_av,x2_err          = fn.dispersion(nconf*n, x2_sum, x4_sum)
    x4_av,x4_err          = fn.dispersion(nconf*n, x4_sum, x8_sum)
    
    Va_av[ialpha]  = valpha_av
    Va_err[ialpha] = valpha_err
    
    if ialpha % (2 * n_alpha) == 0:
        da = dalpha / 4.0
    else:
        da = dalpha / 2.0   #chiedere informazioni
        
    de = da * valpha_av
    ei += de
    #---------outputs----------------------------------------------------------|
    switch.write('alpha\t stot_av\t stot_err\n')
    switch.write("{:.2f}\t{:.4f}\t{:.4f}\n".format(alpha,stot_av,stot_err))
    switch.write('x_av, x_err, x2_av, x2_err, x4_av, x4_err\n')
    switch.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format( x_av, x_err, x2_av, x2_err, x4_av, x4_err))
    switch.write('v_av\t v_err\t valpha_av\t valpha_err\n')
    switch.write("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\n".format( v_av, v_err, valpha_av, valpha_err))
    switch.write('ei\t de\t e0\n')
    switch.write("{:.4f}\t{:.4f}\t{:.4f}\n\n".format( ei, de, e0))
#----------------end of loops over coupling constant alpha---------------------|
#------have sum=1/2(up+down) and up = 1/2*f0+f1+...+1/2*fn, down=..------------|
ei, de_tot = fn.summing(n_alpha, dalpha,e0, Va_av, Va_err)
# -------------------------uncertainties --------------------------------------|                                                                        

#outputs
switch.write('input parameters\n -----------\n')
switch.write('beta\t f0\t e0\n')
switch.write("{:.4f}\t{:.4f}\t{:.4f}\n\n".format( beta, f0, e0))
switch.write('final initial energy\n ei\t de\t e0\n')
switch.write("{:.4f}\t{:.4f}\t{:.4f}\n\n".format( ei, de, e0))
switch.write(' ei\t de_tot\t \n')
switch.write("{:.4f}\t{:.4f}\n".format( ei, de_tot))
switch.close()







                 