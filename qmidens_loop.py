import numpy as np
import functions as fn
import random
from tqdm import tqdm
import os
'''
------------------------------------------------------------------------------
   Lattice calculation in quantum mechanics                                                                         
------------------------------------------------------------------------------
   Action m/2(\dot x)^2+k(x^2-f^2)^2, units 2m=k=1 
------------------------------------------------------------------------------
   Calculate instanton density by adiabatically switching from gaussian
   approximation to full potential. Have d(inst)=d(gaussian)*exp(-S(non-gaussian)) 
   where S(non-gaussian)=\int d\alpha <S-S(gaussian)>_\alpha. Perform
   reference calculation of fluctuations around trivial vacuum.
------------------------------------------------------------------------------
   Instanton is placed at beta/2. anti-symmetric boundary conditions are
   used. position is fixed during update by requiring x(beta/2)=0
------------------------------------------------------------------------------
'''
#   Input parameters 
#------------------------------------------------------------------------------

def setting_inputs():
    f = 1.4 #minimum of the potential
    n = 800 #lattice points
    a = 0.05 #lattice spacing
    neq = 100 #number of equilibration sweeps
    nmc = 10**4 #number of MonteCarlo sweeps
    dx = 0.5 #width of updates
    n_alpha = 20 #number max of points in the correlation functions
    nc = 5 #number of correlator measurements in a configuration
    kp = 50 #number of sweeps between writeout of complete configuration 
    mode = 1 # ih=0: cold start, x_i=-f; ih=1: hot start, x_i=random
    seed = 597
    w0 = 5.6
    return f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0
#-------initialize inputs------------------------------------------------------|
f, n, a, neq, nmc, dx, n_alpha, nc, kp, mode, seed, w0 = setting_inputs()
random.seed(seed)

fn.directory('qmidens')
density = open('Data/qmidens/density.dat', 'w')
density.write('f\t dens_ng\t err\n')

f = np.linspace(0.25, 1.75, num = 10)

for loop in range(10):
    Vainst_av = np.zeros(2*n_alpha+1)
    Vainst_err = np.zeros(2*n_alpha+1)
    Vavac_av = np.zeros(2*n_alpha+1)
    Vavac_err = np.zeros(2*n_alpha+1)
    Vavinst_sum = 0 #Potential
    Vav2inst_sum = 0
    Valphainst_sum = 0 #Potential
    Valpha2inst_sum = 0
    Vavvac_sum = 0 #Potential
    Vav2vac_sum = 0
    Valphavac_sum = 0 #Potential
    Valpha2vac_sum = 0
    sng = 0.0
    svacng = 0.0
    pi     = np.pi
    dalpha = 1.0/float(n_alpha)
    beta   = n*a
    tau0   = beta/2.0
    s0   = 4.0/3.0*f[loop]**3     #classical action for instanton solution
    dens = 8*np.sqrt(2.0/pi)*f[loop]**2.5*np.exp(-s0)   #unperturbed tunneling rate
    f0   = dens
    #------initialize instanton and gaussian potential-----------------------------|
    x_inst = fn.initialize_instanton(n, a, f[loop], tau0)
    x0_inst = np.copy(x_inst) 
    x_vac = fn.initialize_vacuum(n,f[loop])
    x0_vac = np.copy(x_vac)

    w_inst = -4.0*(f[loop]**2-3.0*x0_inst**2)
    v_inst = (x0_inst**2-f[loop]**2)**2 

    w_vac = np.full(n+1, 8 * f[loop]**2)
    v_vac = np.zeros(n+1)
    for ialpha in tqdm(range((2 * n_alpha + 1))):
        if ialpha <= n_alpha:
            alpha = ialpha * dalpha  #da 0 a 1
        else:
            alpha = 2.0 - ialpha * dalpha #da 1 a 0
        nconf = 0
        ncor  = 0
        #---------montecarlo generations-------------------------------------------|
        for i in range(nmc):
            nconf += 1
            if i == neq:
                nconf = 0
                Vavinst_sum = 0 #Potential
                Vav2inst_sum = 0
                Valphainst_sum = 0 #Potential
                Valpha2inst_sum = 0
                Vavvac_sum = 0 #Potential
                Vav2vac_sum = 0
                Valphavac_sum = 0 #Potential
                Valpha2vac_sum = 0
            
            x_inst = fn.update_instanton(x_inst, x0_inst, w_inst, v_inst, n, a, f[loop], alpha, dx)
            x_vac = fn.update_vacuum(x_vac, x0_vac, w_vac, v_vac, n, a, f[loop], alpha, dx)
            V_inst = fn.fluctuations_total_potential(x_inst, x0_inst, w_inst, v_inst, f[loop], a, alpha, n)
            V_vac = fn.fluctuations_total_potential(x_vac, x0_vac, w_vac, v_vac, f[loop], a, alpha, n)
            
            Valpha_inst = fn.delta_V(x_inst, x0_inst, w_inst, v_inst, f[loop], a, n)
            Valpha_vac = fn.delta_V(x_vac, x0_vac, w_vac, v_vac, f[loop], a, n)
            Vavinst_sum  += V_inst/beta
            Vav2inst_sum += V_inst**2/beta
            Valphainst_sum  += Valpha_inst/beta
            Valpha2inst_sum += Valpha_inst**2/beta
            Vavvac_sum  += V_vac/beta
            Vav2vac_sum += V_vac**2/beta
            Valphavac_sum  += Valpha_vac/beta
            Valpha2vac_sum += Valpha_vac**2/beta
            
        vinst_av, vinst_err           = fn.dispersion(nconf, Vavinst_sum, Vav2inst_sum)
        valphainst_av, valphainst_err = fn.dispersion(nconf,Valphainst_sum, Valpha2inst_sum)
           
        Vainst_av[ialpha]  = valphainst_av
        Vainst_err[ialpha] = valphainst_err
            
        vvac_av, vvac_err           = fn.dispersion(nconf, Vavvac_sum, Vav2vac_sum)
        valphavac_av, valphavac_err = fn.dispersion(nconf,Valphavac_sum, Valpha2vac_sum)
           
        Vavac_av[ialpha]  = valphavac_av
        Vavac_err[ialpha] = valphavac_err
            
        if ialpha % (2 * n_alpha) == 0:
            da = dalpha / 4.0
        else:
            da = dalpha / 2.0   #chiedere informazioni
        dsng = da * valphainst_av
        sng += dsng
        dsvacng = da * valphavac_av
        svacng += dsvacng
    #----------------end of loops over coupling constant alpha---------------------|
    #------have sum=1/2(up+down) and up = 1/2*f0+f1+...+1/2*fn, down=..------------|
    supinst_sum, supinst_err, supinst_hal, sdwinst_sum, sdwinst_err, sdwinst_hal = fn.summing(n_alpha, dalpha, Vainst_av, Vainst_err)
    svacup_sum, svacup_err, svacup_hal, svacdw_sum, svacdw_err, svacdw_hal = fn.summing(n_alpha, dalpha, Vainst_av, Vainst_err)
    #   uncertainties                                                                         
    #------------------------------------------------------------------------------
    ds     = supinst_sum + sdwinst_sum
    sng    = ds
    dens_ng= dens*np.exp(-sng)
    ds_err = np.sqrt(supinst_err + sdwinst_err)
    ds_hal = supinst_hal + sdwinst_hal
    ds_dif = abs(supinst_sum - sdwinst_sum)
    ds_dis = abs(ds - ds_hal)/2.0
    ds_tot = np.sqrt(ds_err**2 + ds_dif**2 + ds_dis**2)
    dens_er= dens_ng*ds_tot 
    dsvac     = svacup_sum + svacdw_sum
    svacng    = dsvac
    fvac      = np.exp(-svacng)
    dsvac_err = np.sqrt(svacup_err + svacdw_err)
    dsvac_hal = svacup_hal + svacdw_hal
    dsvac_dif = abs(svacup_sum - svacdw_sum)
    dsvac_dis = abs(dsvac - dsvac_hal)/2.0
    dsvac_tot = np.sqrt(dsvac_err**2 + dsvac_dif**2 + dsvac_dis**2)
    fvac_er   = fvac*dsvac_tot 
    # final answer                                                                          
    #------------------------------------------------------------------------------
    seff    = sng - svacng
    seff_er = np.sqrt(ds_tot**2+dsvac_tot**2)
    dens_ng = dens*np.exp(-seff)
    dens_er = dens_ng*seff_er
    density.write('{:.4f}\t{:.4f}\t{:.4f}\n'.format(f[loop], dens_ng, dens_er))

density.close()
    


