import numpy as np
import functions as fn
import random
from tqdm import tqdm
import inputs
'''
This programme runs in loop qmswitch.py in order to compute 6 values\
    of the free enrgy for differetn values of temperature(inverse of euclidean time)
'''

#--------------------------setting inputs------------------------------------
n, f, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0 = inputs.qmswitch()
random.seed(seed)
#--------------------------output files--------------------------------------
fn.directory('qmswitch')
energy = open('Data/qmswitch/energies.dat', 'w')
energy.write('T\t E\n')  
#---------------------running qmswitch for 6 different beta-------------------      
for index in range(6):
    if index == 0:
        n =  800 
    else:
        n /= 2
        n = int(n)
    #counters
    nconf = 0
    ncor  = 0
    #definitions
    dalpha = 1.0/float(n_alpha)
    beta   = n*a
    e0     = w0/2.0   #energy of ground state of harmonic oscillator
    f0     = 1.0/beta*np.log(2.0*np.sinh(e0*beta)) #harmonic oscillator free energy
    ei     = e0      
    #-------variable definitions----------------------------------------|
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
    for ialpha in tqdm(range(2 * n_alpha )):
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
            x_sum  += np.sum(x)
            x2_sum += np.sum(x**2)
            x4_sum += np.sum(x**4)
            x8_sum += np.sum(x**8)
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
       
    #----------------end of loops over coupling constant alpha---------------------|
    #------have sum=1/2(up+down) and up = 1/2*f0+f1+...+1/2*fn, down=..------------|
    ei, de_tot = fn.summing(n_alpha, dalpha, Va_av, Va_err, e0)
                                                                            
    
    #----------------outputs----------------------------------------------
    energy.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(1/beta, -ei, de_tot))
    print(-ei)
energy.close()
